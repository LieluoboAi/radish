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

#include <type_traits>

#include "torch/data/samplers.h"
#include "torch/nn/module.h"
#include "torch/torch.h"
#include "torch/types.h"

#include "radish/optimization/lamb.h"
#include "radish/optimization/radam.h"
#include "radish/train/data/leveldb_dataset.h"
#include "radish/train/data/txt_dataset.h"
#include "radish/train/model_io.h"
#include "radish/train/progress_reporter.h"
#include "radish/utils/logging.h"
#include "torch/optim/adam.h"

#if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<filesystem>)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#include "ghc/filesystem.hpp"
namespace fs = ghc::filesystem;
#endif

namespace radish {

namespace train {
using Tensor = torch::Tensor;

template <class SampleParser, class Model, bool use_eval_for_best_model = false,
          int64_t maxTrackHist = 8, bool usePlainTxt = true>
class LlbTrainer {
 public:
  LlbTrainer(std::string logdir)
      : logdir_(logdir), best_loss_(1e9), no_best_track_times_(0) {
    best_model_path_ = absl::StrCat(logdir_, "/best_model.ptc");
  }
  virtual ~LlbTrainer() {}
  typedef typename std::conditional<usePlainTxt, data::TxtDataset<SampleParser>,
                                    data::LeveldbDataset<SampleParser>>::type
      DatasetT;

  typedef typename std::conditional<
      usePlainTxt, torch::data::samplers::SequentialSampler,
      torch::data::samplers::RandomSampler>::type DataSamplerT;
  /**
   *
   * 训练主代码
   */
  void MainLoop(Model model, const std::string& trainDatasetPath,
                const std::string& testDatasetPath, double learningRate,
                int batchSize, int64_t evalEvery, ProgressReporter* reporter,
                std::string parserConfPath = {}, int epochs = 50,
                int warmSteps = 1, int64_t maxTestNum = 0,
                int64_t updatePerBatches = 1,
                std::string pretrainModelPath = "",
                std::string pretrainPrefixVarName = "") {
    Json::Value parserConf;
    if (!parserConfPath.empty()) {
      Json::Reader reader;
      std::ifstream ifs(parserConfPath);
      CHECK(ifs) << "can't read " << parserConfPath << " ?";
      CHECK(reader.parse(ifs, parserConf)) << "config file can't be parsed!";
    }
    DatasetT testDataset(testDatasetPath, parserConf);
    std::vector<std::vector<Tensor>> testDatas;
    std::vector<Tensor> testTargets;
    auto testLoader = torch::data::make_data_loader<DataSamplerT>(
        std::move(testDataset),
        torch::data::DataLoaderOptions().batch_size(1).workers(1));
    spdlog::info(
        "try load test dataset into memory,  max test examples allowed to "
        "load={}....",
        maxTestNum > 0 ? std::to_string(maxTestNum) : "unset");
    int ntest = 0;
    for (auto& input : *testLoader) {
      auto& ex = input[0];
      if (ex.features.empty()) {
        continue;
      }
      testDatas.push_back(ex.features);
      testTargets.push_back(ex.target);
      ++ntest;
      if (maxTestNum > 0 && ntest >= maxTestNum) {
        spdlog::info("only allow to load {} test examples!", maxTestNum);
        break;
      }
    }
    spdlog::info("loaded {} test examples!", testDatas.size());
    torch::Device device = torch::kCPU;
    spdlog::info("CUDA DEVICE COUNT: {}", torch::cuda::device_count());
    if (torch::cuda::is_available()) {
      if (torch::cuda::cudnn_is_available()) {
        spdlog::info("CUDA  and cudnn is available! Training on GPU.");
      } else {
        spdlog::info(
            "CUDA is available, but cudnn is not available! Training on GPU.");
      }
      device = torch::kCUDA;
    }

    std::vector<Tensor> paramters;
    std::vector<std::string> names;
    for (auto kv : model->named_parameters()) {
      paramters.push_back(kv.value());
      names.push_back(kv.key());
    }

    radish::optim::RAdam radam(paramters, names,
                               radish::optim::RAdamOptions(learningRate)
                                   .warmup_steps(warmSteps)
                                   .weight_decay(0.01));

    // radish::optim::Lamb radam(
    //     paramters, names,
    //     radish::optim::LambOptions(learningRate).weight_decay(0.01));

    // log目录初始化
    logdir_init_(model, pretrainModelPath, pretrainPrefixVarName);
    model->to(device);
    radam.zero_grad();
    int64_t steps = 0;
    int64_t update_batch = 0;
    std::vector<float> evals;
    // first eval loss on test set
    auto loss_v =
        _run_on_test(model, testDatas, testTargets, batchSize, device, evals);
    reporter->UpdateProgress(0, absl::nullopt, {loss_v}, evals);
    if (use_eval_for_best_model && evals.size() > 0) {
      loss_v = 0 - evals[0];
    }
    best_loss_ = loss_v;
    for (int i = 0; i < epochs; i++) {
      auto trainLoader = torch::data::make_data_loader<DataSamplerT>(
          std::move(DatasetT(trainDatasetPath, parserConf)),
          torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));
      spdlog::info("start epoch:{}", i);
      for (auto inputs : *trainLoader) {
        model->train();
        std::vector<std::vector<Tensor>> batchDatas;
        std::vector<Tensor> batchTargets;
        for (size_t i = 0; i < inputs.size(); i++) {
          auto& ex = inputs[i];
          if (ex.features.empty()) {
            continue;
          }
          batchDatas.push_back(ex.features);
          batchTargets.push_back(ex.target);
        }
        std::vector<Tensor> examples;
        std::vector<Tensor> targets;
        _prepare_bacth_data(batchDatas, batchTargets, 0, batchDatas.size(),
                            examples, targets, device);
        if (examples.empty()) {
          continue;
        }
        steps += 1;
        Tensor logits = model->forward(examples);
        Tensor target = torch::stack({targets}, 0).to(device);
        evals.clear();
        auto loss = model->CalcLoss(examples, logits, evals, target);
        (void)evals;  // suppress warning
        loss.backward();
        update_batch += 1;
        if (update_batch % updatePerBatches == 0) {
          radam.step();
          update_batch = 0;
          radam.zero_grad();
        }
        float train_loss_v = ((Tensor)loss).item().to<float>();
        if (steps % evalEvery == 0) {
          std::vector<float> tevals;
          auto loss_v = _run_on_test(model, testDatas, testTargets, batchSize,
                                     device, tevals);
          reporter->UpdateProgress(steps, train_loss_v, loss_v, tevals);
          if (use_eval_for_best_model && tevals.size() > 0) {
            loss_v = 0 - tevals[0];
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
    spdlog::info("done trainning....");
  }

 private:
  float _run_on_test(Model model,
                     const std::vector<std::vector<Tensor>>& testDatas,
                     const std::vector<Tensor>& testTargets, int batchSize,
                     torch::Device device, std::vector<float>& evals) {
    model->eval();
    torch::NoGradGuard guard;
    float testLoss = 0;
    size_t nbatch = (testDatas.size() - 1) / batchSize + 1;
    size_t actBatch = 0;
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
      if (examples.empty()) {
        continue;
      }
      actBatch += 1;
      Tensor target = torch::stack({targets}, 0).to(device);
      Tensor logits = model->forward(examples);
      std::vector<float> tevals;
      auto tloss = model->CalcLoss(examples, logits, tevals, target, false);
      float tlv = ((Tensor)tloss).item().to<float>();
      testLoss += tlv;
      if (evals.empty()) {
        evals.resize(tevals.size());
        std::fill(evals.begin(), evals.end(), 0);
      }
      for (size_t i = 0; i < tevals.size(); i++) {
        evals[i] += tevals[i];
      }
    }
    for (size_t i = 0; i < evals.size(); i++) {
      evals[i] /= static_cast<float>(actBatch + 0.00001);
    }
    return testLoss / static_cast<float>(actBatch + 0.00001);
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

  bool logdir_init_(Model model, std::string pretrainModelPath,
                    std::string pretrainPrefixVarName) {
    bool loadedPretrain = false;
    if (!pretrainModelPath.empty()) {
      LoadModelEx(model.ptr(), pretrainModelPath, pretrainPrefixVarName);
      loadedPretrain = true;
    }
    if (!fs::exists(logdir_)) {
      fs::create_directory(logdir_);
    } else {
      if (!loadedPretrain && fs::exists(best_model_path_)) {
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
