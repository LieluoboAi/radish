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

#include "radish/transformer/encoder_layer.h"

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "radish/transformer/positionwise_fc.h"

namespace radish {

EncoderLayerOptions::EncoderLayerOptions(int64_t d_model, int64_t d_inner,
                                         int64_t n_head, int64_t d_k,
                                         int64_t d_v, double dropout)
    : d_model_(d_model),
      d_inner_(d_inner),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      dropout_(dropout) {}

EncoderLayerImpl::EncoderLayerImpl(EncoderLayerOptions options_)
    : options(options_) {
  reset();
}

void EncoderLayerImpl::reset() {
  pos_ffn = register_module(
      "pos_ffn", radish::PositionwiseFC(options.d_model(), options.d_inner(),
                                        options.dropout()));
  slf_attn = radish::MultiheadAttention(options.n_head(), options.d_model(),
                                        options.d_k(), options.d_v(),
                                        options.dropout());
  register_module("slf_attn", slf_attn);
}

void EncoderLayerImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::EncoderLayer(dropout=" << options.dropout()
         << ", n_head=" << options.n_head() << ", d_model=" << options.d_model()
         << ", d_inner=" << options.d_inner() << ", d_k=" << options.d_k()
         << ", d_v=" << options.d_v() << ")";
}

std::vector<Tensor> EncoderLayerImpl::forward(const Tensor& enc_input,
                                              const Tensor& non_pad_mask,
                                              const Tensor& slf_attn_mask) {
  std::vector<Tensor> rets =
      slf_attn->forward(enc_input, enc_input, enc_input, slf_attn_mask);
  Tensor& enc_output = rets[0];
  Tensor& enc_slf_attn = rets[1];
  if (non_pad_mask.numel() > 0) {
    enc_output.mul_(non_pad_mask);
  }
  enc_output = pos_ffn.forward(enc_output);
  if (non_pad_mask.numel() > 0) {
    enc_output.mul_(non_pad_mask);
  }
  return {enc_output, enc_slf_attn};
}
}  // namespace radish
