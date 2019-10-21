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

BertEmbeddingImpl::BertEmbeddingImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}

void BertEmbeddingImpl::reset() {
  int emdsz = options.hidden_size();
  if (options.need_factor_embedding()) {
    emdsz = options.d_wordvec();
  }
  word_embeddings =
      Embedding(EmbeddingOptions(options.n_vocab(), emdsz).padding_idx(0));
  register_module("word_embeddings", word_embeddings);
  position_embeddings =
      Embedding(EmbeddingOptions(options.max_pos(), options.hidden_size()));
  register_module("position_embeddings", position_embeddings);
  token_type_embeddings =
      Embedding(EmbeddingOptions(options.max_types(), options.hidden_size()));
  register_module("token_type_embeddings", token_type_embeddings);
  layer_norm = LayerNorm(options.hidden_size(), options.ln_eps());
  register_module("LayerNorm", layer_norm);
  dropout = torch::nn::Dropout(options.dropout());
  register_module("dropout", dropout);
  if (options.need_factor_embedding()) {
    embedding_to_hidden_proj =
        torch::nn::Linear(torch::nn::LinearOptions(emdsz, options.hidden_size())
                              .with_bias(false));
    register_module("embedding_to_hidden_proj", embedding_to_hidden_proj);
  }
  torch::NoGradGuard guard;
  torch::nn::init::normal_(word_embeddings->weight, 0, options.init_range());
  torch::nn::init::normal_(position_embeddings->weight, 0,
                           options.init_range());
  torch::nn::init::normal_(token_type_embeddings->weight, 0,
                           options.init_range());
  if (options.need_factor_embedding()) {
    torch::nn::init::normal_(embedding_to_hidden_proj->weight, 0,
                             options.init_range());
  }
}

/// Pretty prints the `BertEmbedding` module into the given `stream`.
void BertEmbeddingImpl::pretty_print(std::ostream& stream) const {}

Tensor BertEmbeddingImpl::forward(Tensor inputIds, Tensor typeIds,
                                  Tensor posIds) {
  int seqLen = inputIds.size(1);
  if (posIds.numel() == 0) {
    posIds = torch::arange(
        seqLen, torch::TensorOptions(torch::kInt64).device(inputIds.device()));
    posIds = posIds.unsqueeze(0).expand_as(inputIds);
  }
  if (typeIds.numel() == 0) {
    typeIds = torch::zeros_like(inputIds);
  }
  auto output = word_embeddings(inputIds);
  if (options.need_factor_embedding()) {
    output = embedding_to_hidden_proj(output);
  }
  output.add_(position_embeddings(posIds))
               .add_(token_type_embeddings(typeIds));

  output = layer_norm(output);
  output = dropout(output);
  return output;
}

}  // namespace radish
