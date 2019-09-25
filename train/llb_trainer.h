/*
 * File: llb_trainer.h
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-19 5:55:50
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <experimental/filesystem>

#include "torch/nn/module.h"
#include "torch/torch.h"

#include "torch/types.h"

#include "optimization/radam.h"
#include "train/data/leveldb_dataset.h"
#include "train/model_io.h"
#include "train/progress_reporter.h"
#include "utils/logging.h"

namespace radish {

namespace train {
using Tensor = torch::Tensor;
namespace fs = std::experimental::filesystem;

template <class SampleParser, class Model, bool use_eval_for_best_model = false,
          int64_t maxTrackHist = 8, int64_t update_per_batches = 1>
class LlbTrainer {
 public:
  LlbTrainer(std::string logdir)
      : logdir_(logdir), best_loss_(1e9), no_best_track_times_(0) {
    best_model_path_ = absl::StrCat(logdir_, "/best_model.ptc");
  }
  virtual ~LlbTrainer() {}

  /**
   *
   * 训练主代码
   */
  void MainLoop(Model model, const std::string& trainDatasetPath,
                const std::string& testDatasetPath, double learningRate,
                int batchSize, int64_t evalEvery, ProgressReporter* reporter,
                int epochs = 50, int warmSteps = 1) {
    data::LeveldbDataset<SampleParser> trainDataset(trainDatasetPath);
    data::LeveldbDataset<SampleParser> testDataset(testDatasetPath);
    std::vector<std::vector<Tensor>> testDatas;
    std::vector<Tensor> testTargets;
    auto testLoader = torch::data::make_data_loader(
        testDataset, torch::data::DataLoaderOptions().batch_size(1).workers(1));
    spdlog::info("try load test dataset into memory....");
    int ntest = 0;
    for (auto& input : *testLoader) {
      auto& ex = input[0];
      testDatas.push_back(ex.features);
      testTargets.push_back(ex.target);
      ++ntest;
    }
    spdlog::info("loaded {} test examples!", testDatas.size());
    torch::Device device = torch::kCPU;
    spdlog::info("CUDA DEVICE COUNT: {}", torch::cuda::device_count());
    if (torch::cuda::is_available()) {
      spdlog::info("CUDA is available! Training on GPU.");
      device = torch::kCUDA;
    }

    std::vector<Tensor> paramters;
    std::vector<std::string> names;
    for (auto kv : model->named_parameters()) {
      paramters.push_back(kv.value());
      names.push_back(kv.key());
    }

    radish::optim::RAdam radam(
        paramters, names,
        radish::optim::RAdamOptions(learningRate).warmup_steps(warmSteps));

    // log目录初始化
    logdir_init_(model);
    model->to(device);
    radam.zero_grad();
    int64_t steps = 0;
    int64_t update_batch = 0;
    // first eval loss on test set
    auto [loss_v, eval_v] =
        _run_on_test(model, testDatas, testTargets, batchSize, device);
    if (use_eval_for_best_model) {
      loss_v = 0 - eval_v;
    }
    best_loss_ = loss_v;
    reporter->UpdateProgress(0, absl::nullopt, {loss_v}, {eval_v});
    for (int i = 0; i < epochs; i++) {
      auto trainLoader = torch::data::make_data_loader(
          trainDataset,
          torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));

      for (auto inputs : *trainLoader) {
        steps += 1;
        model->train();
        std::vector<std::vector<Tensor>> batchDatas;
        std::vector<Tensor> batchTargets;
        for (size_t i = 0; i < inputs.size(); i++) {
          auto& ex = inputs[i];
          batchDatas.push_back(ex.features);
          batchTargets.push_back(ex.target);
        }
        std::vector<Tensor> examples;
        std::vector<Tensor> targets;
        _prepare_bacth_data(batchDatas, batchTargets, 0, batchDatas.size(),
                            examples, targets, device);

        Tensor logits = model->forward(examples);
        Tensor target = torch::stack({targets}, 0).to(device);
        auto [loss, _] = model->CalcLoss(examples, logits, target);
        loss.backward();
        update_batch += 1;
        if (update_batch % update_per_batches == 0) {
          radam.step();
          update_batch = 0;
          radam.zero_grad();
        }
        float train_loss_v = ((Tensor)loss).item().to<float>();
        if (steps % evalEvery == 0) {
          auto [loss_v, eval_v] =
              _run_on_test(model, testDatas, testTargets, batchSize, device);
          reporter->UpdateProgress(steps, train_loss_v, loss_v, eval_v);
          if (use_eval_for_best_model) {
            loss_v = 0 - eval_v;
          }
          if (loss_v < best_loss_) {
            best_loss_ = loss_v;
            no_best_track_times_ = 0;
            SaveModel(model.ptr(), best_model_path_);
          } else {
            no_best_track_times_ += 1;
            if (no_best_track_times_ > maxTrackHist) {
              spdlog::warn(
                  "always no improment after {} evals, minimal val is:{}!",
                  maxTrackHist, best_loss_);
              break;
            }
          }
        } else {
          reporter->UpdateProgress(steps, train_loss_v, absl::nullopt,
                                   absl::nullopt);
        }
      }
    }
    if (update_batch > 0) {
      radam.step();
      update_batch = 0;
      radam.zero_grad();
    }
  }

 private:
  std::tuple<float, float> _run_on_test(
      Model model, const std::vector<std::vector<Tensor>>& testDatas,
      const std::vector<Tensor>& testTargets, int batchSize,
      torch::Device device) {
    model->eval();
    torch::NoGradGuard guard;
    double testLoss = 0, evalValue = 0;
    size_t nbatch = (testDatas.size() - 1) / batchSize;
    for (size_t b = 0; b < nbatch; b++) {
      size_t off = b * batchSize;
      // 不包含
      size_t end = off + batchSize;
      if (end > testDatas.size()) {
        end = testDatas.size();
      }
      std::vector<Tensor> examples;
      std::vector<Tensor> targets;
      _prepare_bacth_data(testDatas, testTargets, off, end, examples, targets,
                          device);
      Tensor target = torch::stack({targets}, 0).to(device);
      Tensor logits = model->forward(examples);
      auto [tloss, teval] = model->CalcLoss(examples, logits, target, false);
      float tlv = ((Tensor)tloss).item().to<float>();
      float tev = ((Tensor)teval).item().to<float>();
      testLoss += tlv;
      evalValue += tev;
    }
    return {testLoss / static_cast<float>(nbatch + 0.00001),
            evalValue / static_cast<float>(nbatch + 0.00001)};
  }
  void _prepare_bacth_data(const std::vector<std::vector<Tensor>>& testDatas,
                           const std::vector<Tensor>& testTargets, size_t off,
                           size_t end, std::vector<Tensor>& examples,
                           std::vector<Tensor>& targets, torch::Device device) {
    if (off == end || testDatas.size() == 0) {
      return;
    }
    std::vector<std::vector<Tensor>> retFeatures;
    retFeatures.resize(testDatas[0].size());
    for (size_t i = off; i < end; i++) {
      const std::vector<Tensor>& feature = testDatas[i];
      for (size_t k = 0; k < feature.size(); k++) {
        retFeatures[k].push_back(feature[k]);
      }
      targets.push_back(testTargets[i]);
    }
    examples.resize(retFeatures.size());
    for (size_t k = 0; k < retFeatures.size(); k++) {
      // turn [.....]  into [bsz,....]
      examples[k] = torch::stack({retFeatures[k]}, 0).to(device);
    }
  }

  bool logdir_init_(Model model) {
    if (!fs::exists(logdir_)) {
      fs::create_directory(logdir_);
    } else {
      if (fs::exists(best_model_path_)) {
        LoadModel(model.ptr(), best_model_path_);
        spdlog::info("loaded model from :{}!", best_model_path_);
      }
    }
    return true;
  }
  std::string logdir_;
  std::string best_model_path_;
  float best_loss_;
  int64_t no_best_track_times_;
};
}  // namespace train
}  // namespace radish
