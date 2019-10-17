/*
 * File: bert_attention.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 1:59:07
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <cstddef>
#include <vector>

#include "radish/bert/model/bert_options.h"
#include "radish/layers/layer_norm.h"

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = torch::Tensor;

class TORCH_API BertSelfAttentionImpl
    : public torch::nn::Cloneable<BertSelfAttentionImpl> {
 public:
  BertSelfAttentionImpl(int64_t num_embeddings, int64_t embedding_dim)
      : BertSelfAttentionImpl(BertOptions(num_embeddings, embedding_dim)) {
  }
  explicit BertSelfAttentionImpl(const BertOptions& options_);

  void reset() override;

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  std::vector<Tensor> forward(Tensor  hidden_states, Tensor attention_mask={}, Tensor head_mask={});

  BertOptions options;
  torch::nn::Linear  query=nullptr;
  torch::nn::Linear  key=nullptr;
  torch::nn::Linear  value=nullptr;
  torch::nn::Dropout dropout=nullptr;
private:
  Tensor transpose_for_scores(Tensor x);
  int attention_head_size_;
  int all_head_size_;
};
TORCH_MODULE(BertSelfAttention);

class TORCH_API BertSelfOutputImpl
    : public torch::nn::Cloneable<BertSelfOutputImpl> {
 public:
  explicit BertSelfOutputImpl(const BertOptions& options_);

  void reset() override;

  Tensor forward(Tensor hidden_states, Tensor input_tensor);

  BertOptions options;
  torch::nn::Linear dense = nullptr;
  // should register as name  'LayerNorm'
  LayerNorm layer_norm = nullptr;
  torch::nn::Dropout dropout = nullptr;
};

TORCH_MODULE(BertSelfOutput);

class TORCH_API BertAttentionImpl
    : public torch::nn::Cloneable<BertAttentionImpl> {
 public:
  explicit BertAttentionImpl(const BertOptions& options_);

  void reset() override;

  std::vector<Tensor> forward(Tensor hidden_states, Tensor attention_mask = {},
                              Tensor head_mask = {});

  BertOptions options;
  BertSelfAttention self = nullptr;
  BertSelfOutput output = nullptr;
  torch::nn::Dropout dropout = nullptr;
};

TORCH_MODULE(BertAttention);
}  // namespace radish
