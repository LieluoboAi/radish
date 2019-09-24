/*
 * File: positionwise_fc.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 3:26:24
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "transformer/positionwise_fc.h"

#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "layers/layer_norm.h"

namespace radish {

PositionwiseFCOptions::PositionwiseFCOptions(int64_t d_in, int64_t d_hidden,
                                             double dropout)
    : d_in_(d_in), d_hidden_(d_hidden), dropout_(dropout) {}

PositionwiseFCImpl::PositionwiseFCImpl(PositionwiseFCOptions options_)
    : options(options_) {
  reset();
}

void PositionwiseFCImpl::reset() {
  dropout = register_module("dropout", torch::nn::Dropout(options.dropout_));
  in2hidden = register_module(
      "in2hidden", torch::nn::Conv1d(options.d_in_, options.d_hidden_, 1));
  hidden2in = register_module(
      "hidden2in", torch::nn::Conv1d(options.d_hidden_, options.d_in_, 1));
  layernorm =
      register_module("layernorm", LayerNorm(options.d_in_));
}

void PositionwiseFCImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::PositionwiseFC(dropout=" << options.dropout_
         << ", d_in=" << options.d_in_ << ", d_hidden=" << options.d_hidden_
         << ")";
}

Tensor PositionwiseFCImpl::forward(const Tensor& input) {
  const auto residual = input;
  Tensor output = input.transpose(1, 2);
  output = hidden2in.forward(torch::relu(in2hidden.forward(output)));
  output = output.transpose(1, 2);
  output = dropout.forward(output);
  return layernorm.forward(output.add_(residual));
}
}  // namespace radish
