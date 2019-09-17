/*
 * File: scale_product_attention.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 2:28:11
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "transformer/scale_product_attention.h"

#include <torch/nn/modules/dropout.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace knlp {

ScaleProductAttentionOptions::ScaleProductAttentionOptions(double temperature,
                                                           double att_dropout)
    : temperature_(temperature), att_dropout_(att_dropout) {}

ScaleProductAttentionImpl::ScaleProductAttentionImpl(
    ScaleProductAttentionOptions options)
    : options(options) {
  reset();
}

void ScaleProductAttentionImpl::reset() {
  dropout =
      register_module("dropout", torch::nn::Dropout(options.att_dropout_));
}

void ScaleProductAttentionImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::ScaleProductAttention(dropout=" << options.att_dropout_
         << ", temperature=" << options.temperature_ << ")";
}

std::vector<Tensor> ScaleProductAttentionImpl::forward(const Tensor& q,
                                                       const Tensor& k,
                                                       const Tensor& v,
                                                       const Tensor& mask) {
  Tensor attn = ::torch::bmm(q, k.transpose(1, 2));
  attn.div_(options.temperature_);
  if (!mask.numel() == 0) {
    attn.masked_fill_(mask, -INFINITY);
  }
  attn = ::torch::softmax(attn, 2);
  attn = dropout.forward(attn);
  const auto output = torch::bmm(attn, v);
  return {output, attn};
}
}  // namespace knlp
