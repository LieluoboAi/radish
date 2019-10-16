#pragma once

#include <cstddef>
#include <vector>

#include "radish/layers/embedding_layer.h"
#include "radish/layers/layer_norm.h"
#include "torch/nn/cloneable.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = torch::Tensor;
/// Options for the `Embedding` module.
struct TORCH_API BertEmbeddingOptions {
  BertEmbeddingOptions(int64_t n_vocab, int64_t hidden_size)
      : n_vocab_(n_vocab), hidden_size_(hidden_size){};
  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, n_vocab);
  // embedding size
  TORCH_ARG(int64_t, hidden_size);

  // max pos
  TORCH_ARG(int64_t, max_pos);

  // max types
  TORCH_ARG(int64_t, max_types);

  TORCH_ARG(double, ln_eps)=1e-12;
  TORCH_ARG(double, init_range)=0.02;
};

/// Performs a lookup in a fixed size embedding table.
class TORCH_API BertEmbeddingImpl
    : public torch::nn::Cloneable<BertEmbeddingImpl> {
 public:
  BertEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
      : BertEmbeddingImpl(BertEmbeddingOptions(num_embeddings, embedding_dim)) {
  }
  explicit BertEmbeddingImpl(const BertEmbeddingOptions& options_);

  void reset() override;

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Performs a lookup on the embedding table stored in `weight` using the
  /// `indices` supplied and returns the result.
  Tensor forward(const Tensor& indices);

  /// The `Options` used to configure this `BertEmbedding` module.
  /// Changes to `EmbeddingOptions` *after construction* have no effect.
  BertEmbeddingOptions options;

  Embedding word_embeddings;
  Embedding position_embeddings;
  Embedding token_type_embeddings;
  // should register as name  'LayerNorm'
  LayerNorm layer_norm;
  torch::nn::Dropout dropout;
};

}  // namespace radish