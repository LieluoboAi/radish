/*
 * File: layer_norm.cc
 * Project: layers
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 10:51:56
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "layers/layer_norm.h"

#include <torch/types.h>
#include <torch/utils.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace knlp {

LayerNormOptions::LayerNormOptions(torch::IntArrayRef shape, double eps,
                                   bool elementAffine)
    : shape_(shape), eps_(eps), element_affine_(elementAffine) {}

LayerNormImpl::LayerNormImpl(LayerNormOptions options_) : options(options_) {
  reset();
}

void LayerNormImpl::reset() {
  weight = register_parameter("weight", torch::ones(options.shape_));
  bias = register_parameter("bias", torch::zeros(options.shape_));
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LayerNorm(shape=" << options.shape_
         << ", eps=" << options.eps_
         << ", elementwise_affine=" << options.element_affine_ << ")";
}

Tensor LayerNormImpl::forward(const Tensor& input) {
  if (options.element_affine_) {
    return torch::layer_norm(input, options.shape_, weight, bias, options.eps_);
  } else {
    return torch::layer_norm(input, options.shape_, {}, {}, options.eps_);
  }
}
}  // namespace knlp
