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
#include <torch/nn/module.h>
#include <torch/torch.h>

#include <torch/types.h>
#include "train/data/leveldb_dataset.h"
#include "train/progress_reporter.h"

#include "optimization/radam.h"

namespace radish {

namespace train {
using Tensor = torch::Tensor;

template <class SampleParser, class Model>
class LlbTrainer {
 public:
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
    for (auto& input : *testLoader) {
      auto& ex = input[0];
      testDatas.push_back(ex.features);
      testTargets.push_back(ex.target);
    }
    VLOG(0) << "loaded " << testDatas.size() << " test examples!";
    radish::optim::RAdam radam(
        model->parameters(),
        radish::optim::RAdamOptions(learningRate).warmup_steps(warmSteps));
    int64_t steps = 0;
    // first eval loss on test set
    auto [loss_v, eval_v] =
        _run_on_test(model, testDatas, testTargets, batchSize);
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
                            examples, targets);
        radam.zero_grad();
        Tensor logits = model->forward(examples);
        Tensor target = torch::stack({targets}, 0);
        auto [loss, eval] = model->CalcLoss(examples, logits, target);
        loss.backward();
        radam.step();
        float train_loss_v = ((Tensor)loss).item().to<float>();
        if (steps % evalEvery == 0) {
          auto [loss_v, eval_v] =
              _run_on_test(model, testDatas, testTargets, batchSize);
          reporter->UpdateProgress(steps, train_loss_v, loss_v, eval_v);
        } else {
          reporter->UpdateProgress(steps, train_loss_v, absl::nullopt,
                                   absl::nullopt);
        }
      }
    }
  }

 private:
  std::tuple<float, float> _run_on_test(
      Model model, const std::vector<std::vector<Tensor>>& testDatas,
      const std::vector<Tensor>& testTargets, int batchSize) {
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
      _prepare_bacth_data(testDatas, testTargets, off, end, examples, targets);
      Tensor logits = model->forward(examples);
      Tensor target = torch::stack({targets}, 0);
      auto [tloss, teval] = model->CalcLoss(examples, logits, target);
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
                           std::vector<Tensor>& targets) {
    if (off == end || testDatas.size() == 0) {
      return;
    }
    std::vector<std::vector<Tensor>> retFeatures;
    retFeatures.resize(testDatas[0].size());
    for (size_t i = off; i < end; i++) {
      const std::vector<Tensor>& feature = testDatas[i];
      for (size_t k = 0; k < feature.size(); k++) {
        retFeatures[k].push_back(feature[k].unsqueeze(0));
      }
      targets.push_back(testTargets[i].unsqueeze(0));
    }
    examples.resize(retFeatures.size());
    for (size_t k = 0; k < retFeatures.size(); k++) {
      // turn [1,.....]  into [bsz,....]
      examples[k] = torch::stack({retFeatures[k]}, 0);
    }
  }
};
}  // namespace train
}  // namespace radish
