/*
 * File: bert_encoder.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 3:52:40
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "radish/bert/model/bert_attention.h"
#include "radish/bert/model/bert_options.h"
#include "radish/layers/layer_norm.h"

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/container/modulelist.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = torch::Tensor;

class TORCH_API BertEncoderImpl : public torch::nn::Cloneable<BertEncoderImpl> {
 public:
  explicit BertEncoderImpl(const BertOptions& options_);
  void reset() override;
  std::vector<Tensor> forward(Tensor hidden_states, Tensor attention_mask = {},
                              Tensor head_mask = {});

  BertOptions options;
  torch::nn::ModuleList layer = nullptr;

 private:
  std::mt19937 gen_;
};

TORCH_MODULE(BertEncoder);

class TORCH_API BertPoolerImpl : public torch::nn::Cloneable<BertPoolerImpl> {
 public:
  explicit BertPoolerImpl(const BertOptions& options_);
  void reset() override;
  Tensor forward(Tensor hidden_states);

  BertOptions options;
  torch::nn::Linear dense = nullptr;
};

TORCH_MODULE(BertPooler);

}  // namespace radish