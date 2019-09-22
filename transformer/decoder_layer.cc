/*
 * File: encoder_layer.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 6:28:04
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "transformer/decoder_layer.h"

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "transformer/positionwise_fc.h"

namespace radish {

DecoderLayerOptions::DecoderLayerOptions(int64_t d_model, int64_t d_inner,
                                         int64_t n_head, int64_t d_k,
                                         int64_t d_v, double dropout)
    : d_model_(d_model),
      d_inner_(d_inner),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      dropout_(dropout) {}

DecoderLayerImpl::DecoderLayerImpl(DecoderLayerOptions options_)
    : options(options_) {
  reset();
}

void DecoderLayerImpl::reset() {
  pos_ffn = register_module(
      "pos_ffn", radish::PositionwiseFC(options.d_model_, options.d_inner_,
                                        options.dropout_));
  decenc_attn =
      radish::MultiheadAttention(options.n_head_, options.d_model_,
                                 options.d_k_, options.d_v_, options.dropout_);
  slf_attn =
      radish::MultiheadAttention(options.n_head_, options.d_model_,
                                 options.d_k_, options.d_v_, options.dropout_);
  register_module("slf_attn", slf_attn);
  register_module("decenc_attn", decenc_attn);
}

void DecoderLayerImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::DecoderLayer(dropout=" << options.dropout_
         << ", n_head=" << options.n_head_ << ", d_model=" << options.d_model_
         << ", d_inner=" << options.d_inner_ << ", d_k=" << options.d_k_
         << ", d_v=" << options.d_v_ << ")";
}

std::vector<Tensor> DecoderLayerImpl::forward(const Tensor& dec_input,
                                              const Tensor& enc_output,
                                              const Tensor& non_pad_mask,
                                              const Tensor& slf_attn_mask,
                                              const Tensor& dec_enc_attn_mask) {
  std::vector<Tensor> rets =
      slf_attn->forward(dec_input, dec_input, dec_input, slf_attn_mask);
  Tensor dec_output = rets[0];
  Tensor dec_slf_attn = rets[1];
  if (non_pad_mask.numel() > 0) {
    dec_output.mul_(non_pad_mask);
  }
  rets = decenc_attn->forward(dec_output, enc_output, enc_output,
                              dec_enc_attn_mask);
  dec_output = rets[0];
  Tensor dec_enc_attn = rets[1];
  if (non_pad_mask.numel() > 0) {
    dec_output.mul_(non_pad_mask);
  }
  dec_output = pos_ffn.forward(dec_output);
  if (non_pad_mask.numel() > 0) {
    dec_output.mul_(non_pad_mask);
  }
  return {dec_output, dec_slf_attn, dec_enc_attn};
}
}  // namespace radish
