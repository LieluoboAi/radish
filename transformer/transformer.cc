/*
 * File: transformer.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-17 3:46:01
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "transformer/transformer.h"

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace knlp {

TransformerOptions::TransformerOptions(int64_t n_src_vocab, int64_t n_tgt_vocab,
                                       int64_t len_max_seq, int64_t d_word_vec,
                                       int64_t d_model, int64_t d_inner,
                                       int64_t n_layers, int64_t n_head,
                                       int64_t d_k, int64_t d_v, double dropout,
                                       bool tgt_emb_prj_weight_sharing,
                                       bool emb_src_tgt_weight_sharing)
    : n_src_vocab_(n_src_vocab),
      n_tgt_vocab_(n_tgt_vocab),
      len_max_seq_(len_max_seq),
      d_word_vec_(d_word_vec),
      d_model_(d_model),
      d_inner_(d_inner),
      n_layers_(n_layers),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      dropout_(dropout),
      tgt_emb_prj_weight_sharing_(tgt_emb_prj_weight_sharing),
      emb_src_tgt_weight_sharing_(emb_src_tgt_weight_sharing) {}

TransformerImpl::TransformerImpl(TransformerOptions options)
    : options(options) {
  reset();
}

void TransformerImpl::reset() {
  auto topt = torch::nn::LinearOptions(options.d_model_, options.n_tgt_vocab_);
  topt.with_bias_ = false;
  tgt_word_prj = torch::nn::Linear(topt);
  register_module("tgt_word_prj", tgt_word_prj);

  encoder = TransformerEncoder(
      options.n_src_vocab_, options.len_max_seq_, options.d_word_vec_,
      options.n_layers_, options.n_head_, options.d_k_, options.d_v_,
      options.d_model_, options.d_inner_, options.dropout_);
  register_module("encoder", encoder);

  decoder = TransformerDecoder(
      options.n_tgt_vocab_, options.len_max_seq_, options.d_word_vec_,
      options.n_layers_, options.n_head_, options.d_k_, options.d_v_,
      options.d_model_, options.d_inner_, options.dropout_);
  register_module("decoder", decoder);

  CHECK_EQ(options.d_model_, options.d_word_vec_)
      << "To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.";

  if (options.tgt_emb_prj_weight_sharing_) {
    //  Share the weight matrix between target word embedding & the final logit
    //  dense layer
    tgt_word_prj->weight = decoder->tgt_word_emb->weight;
    x_logit_scale = 1.0 / std::sqrt(options.d_model_);
  } else {
    torch::NoGradGuard guard;
    torch::nn::init::xavier_normal_(tgt_word_prj->weight);
  }

  if (options.emb_src_tgt_weight_sharing_) {
    // Share the weight matrix between source & target word embeddings
    CHECK_EQ(options.n_src_vocab_, options.n_tgt_vocab_)
        << "To share word embedding table, the vocabulary size of src/tgt "
           "shall be the same.";
    encoder->src_word_emb->weight = decoder->tgt_word_emb->weight;
  }
}

void TransformerImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::Transformer(dropout=" << options.dropout_
         << ", n_head=" << options.n_head_ << ", d_model=" << options.d_model_
         << ", d_inner=" << options.d_inner_ << ", d_k=" << options.d_k_
         << ", d_v=" << options.d_v_ << ", n_src_vocab=" << options.n_src_vocab_
         << ", n_tgt_vocab=" << options.n_tgt_vocab_
         << ", len_max_seq=" << options.len_max_seq_
         << ", d_word_vec=" << options.d_word_vec_
         << ", n_layers=" << options.n_layers_
         << ", tgt_emb_prj_weight_sharing="
         << options.tgt_emb_prj_weight_sharing_
         << ", emb_src_tgt_weight_sharing="
         << options.emb_src_tgt_weight_sharing_ << ")";
}

Tensor TransformerImpl::forward(const Tensor& src_seq, const Tensor& src_pos,
                                const Tensor& tgt_seq, const Tensor& tgt_pos) {
  std::vector<Tensor> encodeRets = encoder(src_seq, src_pos);
  std::vector<Tensor> decRets =
      decoder(tgt_seq, tgt_pos, src_seq, encodeRets[0]);
  auto seq_logit = tgt_word_prj(decRets[0]).mul(x_logit_scale);
  return seq_logit;
}

}  // namespace knlp
