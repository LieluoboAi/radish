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

#include "transformer/transformer_decoder.h"

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace knlp {

TransformerDecoderOptions::TransformerDecoderOptions(
    int64_t n_tgt_vocab, int64_t len_max_seq, int64_t d_word_vec,
    int64_t n_layers, int64_t n_head, int64_t d_k, int64_t d_v, int64_t d_model,
    int64_t d_inner, double dropout)
    : n_tgt_vocab_(n_tgt_vocab),
      len_max_seq_(len_max_seq),
      d_word_vec_(d_word_vec),
      n_layers_(n_layers),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      d_model_(d_model),
      d_inner_(d_inner),
      dropout_(dropout) {}

TransformerDecoderImpl::TransformerDecoderImpl(
    TransformerDecoderOptions options_)
    : options(options_) {
  reset();
}

void TransformerDecoderImpl::reset() {
  int64_t n_position = options.len_max_seq_ + 1;
  tgt_word_emb = knlp::Embedding(options.n_tgt_vocab_, options.d_word_vec_);
  register_module("tgt_word_emb", tgt_word_emb);
  pos_emb = knlp::Embedding(n_position, options.d_word_vec_);
  register_module("pos_emb", pos_emb);
  for (auto i = 0; i < options.n_layers_; i++) {
    auto layer =
        knlp::DecoderLayer(options.d_model_, options.d_inner_, options.n_head_,
                           options.d_k_, options.d_v_, options.dropout_);
    register_module(std::string("decoder_layer_") + std::to_string(i + 1),
                    layer);
    decoder_stack->push_back(layer);
  }
  torch::NoGradGuard guard;
  torch::nn::init::uniform_(tgt_word_emb->weight, -0.02, 0.02);
  torch::nn::init::uniform_(pos_emb->weight, -0.02, 0.02);
}

void TransformerDecoderImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::TransformerDecoder(dropout=" << options.dropout_
         << ", n_head=" << options.n_head_ << ", d_model=" << options.d_model_
         << ", d_inner=" << options.d_inner_ << ", d_k=" << options.d_k_
         << ", d_v=" << options.d_v_ << ", n_tgt_vocab=" << options.n_tgt_vocab_
         << ", len_max_seq=" << options.len_max_seq_
         << ", d_word_vec=" << options.d_word_vec_
         << ", n_layers=" << options.n_layers_ << ")";
}

static Tensor get_attn_key_pad_mask(const Tensor& seq_k, const Tensor& seq_q) {
  //  Expand to fit the shape of key query attention matrix.
  int64_t len_q = seq_q.size(1);
  Tensor padding_mask = seq_k.eq(0);
  padding_mask = padding_mask.unsqueeze(1).expand({-1, len_q, -1});
  return padding_mask;
}

static Tensor get_non_pad_mask(const Tensor& seq) {
  CHECK_EQ(seq.ndimension(), 2);
  return seq.ne(0).toType(c10::ScalarType::Float).unsqueeze(-1);
}
static Tensor get_subsequent_mask(const Tensor& seq) {
  //  For masking out the subsequent info.
  CHECK_EQ(seq.dim(), 2);
  int64_t sz_b = seq.size(0);
  int64_t len_s = seq.size(1);
  torch::TensorOptions topt(torch::ScalarType::QUInt8);
  topt = topt.device(seq.device());
  Tensor subsequent_mask = torch::triu(torch::ones({len_s, len_s}, topt), 1);
  subsequent_mask =
      subsequent_mask.unsqueeze(0).expand({sz_b, -1, -1});  // b x ls x ls
  return subsequent_mask;
}

std::vector<Tensor> TransformerDecoderImpl::forward(const Tensor& tgt_seq,
                                                    const Tensor& tgt_pos,
                                                    const Tensor& src_seq,
                                                    const Tensor& enc_output,
                                                    bool return_attns) {
  std::vector<Tensor> dec_slf_attn_list;
  std::vector<Tensor> dec_enc_attn_list;
  // -- Prepare masks
  Tensor non_pad_mask = get_non_pad_mask(tgt_seq);
  Tensor slf_attn_mask_subseq = get_subsequent_mask(tgt_seq);
  Tensor slf_attn_mask_keypad = get_attn_key_pad_mask(tgt_seq, tgt_seq);
  Tensor slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0);

  Tensor dec_enc_attn_mask = get_attn_key_pad_mask(src_seq, tgt_seq);

  // -- Forward
  Tensor dec_output = tgt_word_emb->forward(tgt_seq) + pos_emb(tgt_pos);
  for (auto i = 0; i < options.n_layers_; i++) {
    auto elayer = decoder_stack->ptr<DecoderLayerImpl>(i);
    std::vector<Tensor> rets = elayer->forward(
        dec_output, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask);
    dec_output = rets[0];
    const auto dec_slf_attn = rets[1];
    const auto dec_enc_attn = rets[2];
    if (return_attns) {
      dec_slf_attn_list.push_back(dec_slf_attn);
      dec_enc_attn_list.push_back(dec_enc_attn);
    }
  }
  if (return_attns) {
    std::vector<Tensor> todos;
    todos.push_back(dec_output);
    todos.insert(todos.end(), dec_slf_attn_list.begin(),
                 dec_slf_attn_list.end());
    todos.insert(todos.end(), dec_enc_attn_list.begin(),
                 dec_enc_attn_list.end());
    return todos;
  } else {
    return {dec_output};
  }
}
}  // namespace knlp
