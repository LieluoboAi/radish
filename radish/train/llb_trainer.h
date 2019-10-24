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
#include "radish/train/benchmark_submiter.h"
#include "radish/train/data/leveldb_dataset.h"
#include "radish/train/data/txt_dataset.h"
#include "radish/train/model_io.h"
#include "radish/train/progress_reporter.h"
#include "radish/utils/logging.h"
#include "torch/optim/adam.h"

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

  void Benchmark(Model model, const std::string& datasetPath, int batchSize,
                 BenchmarkSubmiter* submiter, std::string parserConfPath) {
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
    // log目录初始化
    logdir_init_(model, std::string(), device);
    model->to(device);
    Json::Value parserConf;
    if (!parserConfPath.empty()) {
      Json::Reader reader;
      std::ifstream ifs(parserConfPath);
      CHECK(ifs) << "can't read " << parserConfPath << " ?";
      CHECK(reader.parse(ifs, parserConf)) << "config file can't be parsed!";
    }
    auto trainLoader = torch::data::make_data_loader<DataSamplerT>(
        std::move(DatasetT(datasetPath, parserConf)),
        torch::data::DataLoaderOptions()
            .batch_size(batchSize)
            .workers(1)
            .enforce_ordering(true));
    int nexs = 0;
    for (auto inputs : *trainLoader) {
      model->eval();
      std::vector<std::vector<Tensor>> batchDatas;
      for (size_t i = 0; i < inputs.size(); i++) {
        auto& ex = inputs[i];
        if (ex.features.empty()) {
          continue;
        }
        if (batchDatas.empty()) {
          batchDatas.resize(ex.features.size());
        } else {
          CHECK_EQ(batchDatas.size(), ex.features.size());
        }
        for (size_t j = 0; j < ex.features.size(); j++) {
          batchDatas[j].push_back(ex.features[j]);
        }
      }
      if (batchDatas.empty()) {
        continue;
      }
      std::vector<Tensor> examples;
      for (size_t j = 0; j < batchDatas.size(); j++) {
        examples.push_back(torch::stack(batchDatas[j], 0).to(device));
      }
      Tensor logits = model->Benchmark(examples);
      CHECK_EQ(logits.dim(), 2);
      for (int i = 0; i < logits.size(0); i++) {
        std::vector<float> row;
        for (int j = 0; j < logits.size(1); j++) {
          row.push_back(logits[i][i].item().to<float>());
        }
        submiter->SubmitOneRow(row);
        nexs += 1;
      }
    }
    submiter->SubmitDone();
    spdlog::info("benchmarked {} exampes", nexs);
  }

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
                std::string pretrainModelPath = "") {
    Json::Value parserConf;
    if (!parserConfPath.empty()) {
      Json::Reader reader;
      std::ifstream ifs(parserConfPath);
      CHECK(ifs) << "can't read " << parserConfPath << " ?";
      CHECK(reader.parse(ifs, parserConf)) << "config file can't be parsed!";
    }
    DatasetT testDataset(testDatasetPath, parserConf);
    std::vector<Tensor> all_test_examples;
    Tensor all_test_targets;
    {
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
        if (testDatas.empty()) {
          testDatas.resize(ex.features.size());
        } else {
          CHECK_EQ(testDatas.size(), ex.features.size());
        }
        for (size_t i = 0; i < testDatas.size(); i++) {
          testDatas[i].push_back(ex.features[i]);
        }
        testTargets.push_back(ex.target);
        ++ntest;
        if (maxTestNum > 0 && ntest >= maxTestNum) {
          spdlog::info("only allow to load {} test examples!", maxTestNum);
          break;
        }
      }
      for (size_t i = 0; i < testDatas.size(); i++) {
        Tensor t = torch::stack(testDatas[i], 0);
        all_test_examples.push_back(t);
      }
      all_test_targets = torch::stack(testTargets, 0);
    }
    spdlog::info("loaded {} test examples!", all_test_targets.size(0));
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
    logdir_init_(model, pretrainModelPath, device);
    model->to(device);
    radam.zero_grad();
    int64_t steps = 0;
    int64_t update_batch = 0;
    std::vector<float> evals;
    // first eval loss on test set
    auto loss_v = _run_on_test(model, all_test_examples, all_test_targets,
                               batchSize, device, evals);
    reporter->UpdateProgress(0, absl::nullopt, {loss_v}, evals);
    if (use_eval_for_best_model && evals.size() > 0) {
      loss_v = 0 - evals[0];
    }
    best_loss_ = loss_v;
    bool earlyReturn = false;
    for (int e = 0; e < epochs; e++) {
      auto trainLoader = torch::data::make_data_loader<DataSamplerT>(
          std::move(DatasetT(trainDatasetPath, parserConf)),
          torch::data::DataLoaderOptions()
              .batch_size(batchSize)
              .workers(2)
              .enforce_ordering(false));
      spdlog::info("start epoch:{}", e);
      for (auto inputs : *trainLoader) {
        model->train();
        std::vector<std::vector<Tensor>> batchDatas;
        std::vector<Tensor> batchTargets;
        for (size_t i = 0; i < inputs.size(); i++) {
          auto& ex = inputs[i];
          if (ex.features.empty()) {
            continue;
          }
          if (batchDatas.empty()) {
            batchDatas.resize(ex.features.size());
          } else {
            CHECK_EQ(batchDatas.size(), ex.features.size());
          }
          for (size_t j = 0; j < ex.features.size(); j++) {
            batchDatas[j].push_back(ex.features[j]);
          }
          batchTargets.push_back(ex.target);
        }
        if (batchTargets.empty()) {
          continue;
        }
        Tensor target = torch::stack(batchTargets, 0).to(device);
        std::vector<Tensor> examples;
        for (size_t j = 0; j < batchDatas.size(); j++) {
          examples.push_back(torch::stack(batchDatas[j], 0).to(device));
        }
        steps += 1;
        std::vector<Tensor> logits = model->forward(examples);
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
          auto loss_v = _run_on_test(model, all_test_examples, all_test_targets,
                                     batchSize, device, tevals);
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
              earlyReturn = true;
              break;
            }
          }
        } else {
          reporter->UpdateProgress(steps, train_loss_v, absl::nullopt,
                                   absl::nullopt);
        }
      }
      if (earlyReturn) {
        break;
      }
    }
    if (update_batch > 0) {
      radam.step();
      update_batch = 0;
      radam.zero_grad();
    }
    spdlog::info("done trainning,  early return ? = {}....", earlyReturn);
  }

 private:
  Tensor select_range_(const Tensor& t, int off, int end) {
    std::vector<Tensor> ts;
    for (int i = off; i < end; i++) {
      ts.push_back(t.select(0, i));
    }
    return torch::stack(ts, 0);
  }
  float _run_on_test(Model model, const std::vector<Tensor>& testDatas,
                     const Tensor& testTargets, int batchSize,
                     torch::Device device, std::vector<float>& evals) {
    model->eval();
    torch::NoGradGuard guard;
    float testLoss = 0;
    if (testDatas.empty()) {
      return 0;
    }
    size_t total = testDatas[0].size(0);
    size_t nbatch = (total - 1) / batchSize + 1;
    size_t actBatch = 0;
    std::vector<std::vector<Tensor>> all_logits;
    evals.clear();
    bool inbatch = model->EvalInBatch();
    for (size_t b = 0; b < nbatch; b++) {
      size_t off = b * batchSize;
      // 不包含
      size_t end = off + batchSize;
      if (end > total) {
        end = total;
      }
      if (off == end) {
        continue;
      }
      std::vector<Tensor> examples;
      Tensor targets = select_range_(testTargets, off, end).to(device);
      for (size_t i = 0; i < testDatas.size(); i++) {
        examples.push_back(select_range_(testDatas[i], off, end).to(device));
      }
      actBatch += 1;
      std::vector<Tensor> logits = model->forward(examples);
      if (inbatch) {
        std::vector<float> tevals;
        auto tloss = model->CalcLoss(examples, logits, tevals, targets);
        if (evals.empty()) {
          evals.insert(evals.begin(), tevals.begin(), tevals.end());
        } else {
          CHECK_EQ(tevals.size(), evals.size());
          for (size_t i = 0; i < tevals.size(); i++) {
            evals[i] += tevals[i];
          }
        }
        testLoss += ((Tensor)tloss).item().to<float>();
      } else {
        if (all_logits.empty()) {
          all_logits.resize(logits.size());
        } else {
          CHECK_EQ(all_logits.size(), logits.size());
        }
        for (size_t i = 0; i < logits.size(); i++) {
          all_logits[i].push_back(logits[i].to(torch::kCPU));
        }
      }
    }  // end for batch

    if (inbatch) {
      for (size_t i = 0; i < evals.size(); i++) {
        evals[i] /= static_cast<float>(actBatch + 1e-10);
      }
      testLoss = testLoss / static_cast<float>(actBatch + 1e-10);
    } else {
      std::vector<Tensor> packed_logits;
      for (size_t i = 0; i < all_logits.size(); i++) {
        Tensor logits = torch::cat(all_logits[i], 0);
        CHECK_EQ(logits.size(0), static_cast<int>(total));
        packed_logits.push_back(logits);
      }
      auto tloss =
          model->CalcLoss(testDatas, packed_logits, evals, testTargets);
      testLoss = ((Tensor)tloss).item().to<float>();
    }
    return testLoss;
  }

  bool logdir_init_(Model model, std::string pretrainModelPath,
                    torch::Device device) {
    bool loadedPretrain = false;
    if (!pretrainModelPath.empty()) {
      if (model->LoadFromPretrain(pretrainModelPath)) {
        loadedPretrain = true;
      } else {
        spdlog::info("try load from :{} error, skip!", pretrainModelPath);
      }
    }
    if (!fs::exists(logdir_)) {
      fs::create_directory(logdir_);
    } else {
      if (!loadedPretrain && fs::exists(best_model_path_)) {
        LoadModel(model.ptr(), best_model_path_, "", device);
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
