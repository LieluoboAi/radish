/*
 * File: transformer_encoder.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-17 10:32:17
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "radish/transformer/transformer_encoder.h"

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "radish/utils/logging.h"

namespace radish {

TransformerEncoderOptions::TransformerEncoderOptions(
    int64_t n_src_vocab, int64_t len_max_seq, int64_t d_word_vec,
    int64_t n_layers, int64_t n_head, int64_t d_k, int64_t d_v, int64_t d_model,
    int64_t d_inner, double dropout)
    : n_src_vocab_(n_src_vocab),
      len_max_seq_(len_max_seq),
      d_word_vec_(d_word_vec),
      n_layers_(n_layers),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      d_model_(d_model),
      d_inner_(d_inner),
      dropout_(dropout) {}

TransformerEncoderImpl::TransformerEncoderImpl(
    TransformerEncoderOptions options_)
    : options(options_) {
  reset();
}

static Tensor get_attn_key_pad_mask(const Tensor& seq_k, const Tensor& seq_q) {
  //  Expand to fit the shape of key query attention matrix.
  int64_t len_q = seq_q.size(1);
  Tensor padding_mask = seq_k.eq(0);
  padding_mask = padding_mask.unsqueeze(1).expand({-1, len_q, -1});
  return padding_mask;
}

static Tensor get_non_pad_mask(const Tensor& seq) {
  CHECK_EQ(seq.dim(), 2);
  return seq.ne(0).toType(torch::kFloat32).unsqueeze(-1);
}

void TransformerEncoderImpl::reset() {
  int64_t n_position = options.len_max_seq() + 1;
  src_word_emb = torch::nn::Embedding(
      torch::nn::EmbeddingOptions(options.n_src_vocab(), options.d_word_vec()));
  register_module("src_word_emb", src_word_emb);
  pos_emb = torch::nn::Embedding(
      torch::nn::EmbeddingOptions(n_position, options.d_word_vec()));
  register_module("pos_emb", pos_emb);
  type_emb = torch::nn::Embedding(
      torch::nn::EmbeddingOptions(options.max_types(), options.d_word_vec()));
  register_module("type_emb", type_emb);
  if (options.need_factor_embedding()) {
    embedding_to_hidden_proj = torch::nn::Linear(
        torch::nn::LinearOptions(options.d_word_vec(), options.d_model())
            .with_bias(false));
    register_module("embedding_to_hidden_proj", embedding_to_hidden_proj);
  }
  for (auto i = 0; i < options.n_layers(); i++) {
    auto layer = radish::EncoderLayer(options.d_model(), options.d_inner(),
                                      options.n_head(), options.d_k(),
                                      options.d_v(), options.dropout());
    register_module(std::string("encoder_layer_") + std::to_string(i + 1),
                    layer);
    encoder_stack->push_back(layer);
  }
  torch::NoGradGuard guard;
  torch::nn::init::uniform_(src_word_emb->weight, -0.02, 0.02);
  torch::nn::init::uniform_(pos_emb->weight, -0.02, 0.02);
  torch::nn::init::uniform_(type_emb->weight, -0.02, 0.02);
  if (embedding_to_hidden_proj) {
    torch::nn::init::xavier_normal_(embedding_to_hidden_proj->weight);
  }
}

void TransformerEncoderImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::TransformerEncoder(dropout=" << options.dropout()
         << ", n_head=" << options.n_head() << ", d_model=" << options.d_model()
         << ", d_inner=" << options.d_inner() << ", d_k=" << options.d_k()
         << ", d_v=" << options.d_v()
         << ", n_src_vocab=" << options.n_src_vocab()
         << ", len_max_seq=" << options.len_max_seq()
         << ", d_word_vec=" << options.d_word_vec()
         << ", max_types=" << options.max_types()
         << ", n_layers=" << options.n_layers() << ")";
}

std::vector<Tensor> TransformerEncoderImpl::forward(const Tensor& src_seq,
                                                    const Tensor& src_pos,
                                                    const Tensor& types,
                                                    bool return_attns) {
  std::vector<Tensor> enc_slf_attn_list;
  CHECK_EQ(src_seq.dim(), 2);
  CHECK_EQ(src_seq.sizes(), src_pos.sizes());

  // -- Prepare masks
  Tensor slf_attn_mask = get_attn_key_pad_mask(src_seq, src_seq);
  Tensor non_pad_mask = get_non_pad_mask(src_seq);
  // # -- Forward
  Tensor enc_output = src_word_emb->forward(src_seq);
  enc_output.add_(pos_emb->forward(src_pos));
  if (types.numel() > 0) {
    CHECK_EQ(src_seq.sizes(), types.sizes());
    enc_output.add_(type_emb->forward(types));
  }
  if (options.need_factor_embedding()) {
    enc_output = embedding_to_hidden_proj(enc_output);
  }
  for (auto i = 0; i < options.n_layers(); i++) {
    auto elayer = encoder_stack->ptr<EncoderLayerImpl>(i);
    std::vector<Tensor> rets =
        elayer->forward(enc_output, non_pad_mask, slf_attn_mask);
    enc_output = rets[0];
    const auto& enc_slf_attn = rets[1];
    if (return_attns) {
      enc_slf_attn_list.push_back(enc_slf_attn);
    }
  }
  if (return_attns) {
    enc_slf_attn_list.insert(enc_slf_attn_list.begin(), enc_output);
  } else {
    enc_slf_attn_list.push_back(enc_output);
  }
  return enc_slf_attn_list;
}
}  // namespace radish
