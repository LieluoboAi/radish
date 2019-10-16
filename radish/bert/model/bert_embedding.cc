/*
 * File: bert_embedding.cc
 * Project: model
 * File Created: Wednesday, 16th October 2019 7:33:01 pm
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Wednesday, 16th October 2019 7:52:09 pm
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include "radish/bert/model/bert_embedding.h"

#include "torch/nn/init.h"

namespace radish {

BertEmbeddingImpl::BertEmbeddingImpl(const BertEmbeddingOptions& options_)
    : options(options_) {
  reset();
}

void BertEmbeddingImpl::reset() {
  word_embeddings =
      Embedding(EmbeddingOptions(options.n_vocab(), options.hidden_size())
                    .padding_idx(0));
  register_module("word_embeddings", word_embeddings);
  position_embeddings =
      Embedding(EmbeddingOptions(options.max_pos(), options.hidden_size()));
  register_module("position_embeddings", position_embeddings);
  token_type_embeddings =
      Embedding(EmbeddingOptions(options.max_types(), options.hidden_size()));
  register_module("token_type_embeddings", token_type_embeddings);
  layer_norm = LayerNorm(options.hidden_size(), options.ln_eps());
  register_module("LayerNorm", layer_norm);
  dropout = torch::nn::Dropout(options, dropout());
  register_module("dropout", dropout);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(word_embeddings->weight, 0, options.init_range());
  torch::nn::init::normal_(position_embeddings->weight, 0,
                           options.init_range());
  torch::nn::init::normal_(token_type_embeddings->weight, 0,
                           options.init_range());
}

/// Pretty prints the `Embedding` module into the given `stream`.
void BertEmbeddingImpl::pretty_print(std::ostream& stream) const {}

/// Performs a lookup on the embedding table stored in `weight` using the
/// `indices` supplied and returns the result.
Tensor BertEmbeddingImpl::forward(const Tensor& indices) { return {}; }

}  // namespace radish
