/*
 * File: bert_layer.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 2:24:50
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <cstddef>
#include <vector>

#include "radish/bert/model/bert_attention.h"
#include "radish/bert/model/bert_options.h"
#include "radish/layers/layer_norm.h"

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = torch::Tensor;

class TORCH_API BertIntermediateImpl
    : public torch::nn::Cloneable<BertIntermediateImpl> {
 public:
  explicit BertIntermediateImpl(const BertOptions& options_);
  void reset() override;
  Tensor forward(Tensor hidden_states);

  BertOptions options;
  torch::nn::Linear dense = nullptr;
};

TORCH_MODULE(BertIntermediate);

class TORCH_API BertOutputImpl : public torch::nn::Cloneable<BertOutputImpl> {
 public:
  explicit BertOutputImpl(const BertOptions& options_);

  void reset() override;

  Tensor forward(Tensor hidden_states, Tensor input_tensor);

  BertOptions options;
  torch::nn::Linear dense = nullptr;
  // should register as name  'LayerNorm'
  LayerNorm layer_norm = nullptr;
  torch::nn::Dropout dropout = nullptr;
};

TORCH_MODULE(BertOutput);

class TORCH_API BertLayerImpl : public torch::nn::Cloneable<BertLayerImpl> {
 public:
  explicit BertLayerImpl(const BertOptions& options_);

  void reset() override;

  std::vector<Tensor> forward(Tensor hidden_states, Tensor attention_mask = {},
                 Tensor head_mask = {});

  BertOptions options;
  BertAttention attention = nullptr;
  BertIntermediate intermediate = nullptr;
  BertOutput output = nullptr;
};

TORCH_MODULE(BertLayer);

}  // namespace radish
