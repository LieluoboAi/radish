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

#include "torch/types.h"
#include "torch/utils.h"

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "utils/logging.h"

namespace radish {

LayerNormOptions::LayerNormOptions(int lastDim, double eps, bool elementAffine)
    : last_dim_(lastDim), eps_(eps), element_affine_(elementAffine) {}

LayerNormImpl::LayerNormImpl(LayerNormOptions options_) : options(options_) {
  reset();
}

void LayerNormImpl::reset() {
  weight = register_parameter("weight", torch::ones({options.last_dim_}));
  bias = register_parameter("bias", torch::zeros({options.last_dim_}));
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LayerNorm(shape=" << options.last_dim_
         << ", eps=" << options.eps_
         << ", elementwise_affine=" << options.element_affine_ << ")";
}

Tensor LayerNormImpl::forward(const Tensor& input) {
  if (options.element_affine_) {
    return torch::layer_norm(input, {options.last_dim_}, weight, bias,
                             options.eps_);
  } else {
    return torch::layer_norm(input, {options.last_dim_}, {}, {}, options.eps_);
  }
}
}  // namespace radish
