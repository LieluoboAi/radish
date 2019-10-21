/*
 * File: bert_classification_model.h
 * Project: finetune
 * File Created: Sunday, 20th October 2019 9:01:17 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 9:01:20 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#pragma once

#include "radish/bert/model/bert_model.h"
#include "radish/train/llb_model.h"
namespace radish {
using Tensor = torch::Tensor;

class TORCH_API BertClassificationModelImpl : public train::LlbModel {
 public:
  BertClassificationModelImpl(BertOptions options, int n_class);

  Tensor CalcLoss(const std::vector<Tensor> &examples, const std::vector<Tensor> &logits,
                  std::vector<float> &evals, const Tensor &target = {}) override;

  std::vector<Tensor> forward(std::vector<Tensor> inputs) override;
  bool LoadFromPretrain(std::string path) override;
  BertOptions options;
  int n_class;
  BertModel bert = nullptr;
  torch::nn::Linear final_proj = nullptr;
};

TORCH_MODULE(BertClassificationModel);
}  // namespace radish
