/*
 * File: bert_embedding.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 9:57:40
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <cstddef>
#include <vector>

#include "radish/layers/embedding_layer.h"
#include "radish/layers/layer_norm.h"
#include "radish/bert/model/bert_options.h"

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = torch::Tensor;

/// Performs a lookup in a fixed size embedding table.
class TORCH_API BertEmbeddingImpl
    : public torch::nn::Cloneable<BertEmbeddingImpl> {
 public:
  BertEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
      : BertEmbeddingImpl(BertOptions(num_embeddings, embedding_dim)) {
  }
  explicit BertEmbeddingImpl(const BertOptions& options_);

  void reset() override;

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(Tensor inputIds, Tensor typeIds = {}, Tensor posIds = {});

  BertOptions options;

  Embedding word_embeddings = nullptr;
  Embedding position_embeddings = nullptr;
  Embedding token_type_embeddings = nullptr;
  // should register as name  'LayerNorm'
  LayerNorm layer_norm = nullptr;
  torch::nn::Dropout dropout = nullptr;
};

TORCH_MODULE(BertEmbedding);
}  // namespace radish