/*
 * File: embedding.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-15 5:41:59
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "layers/embedding.h"

#include <torch/types.h>
#include <torch/utils.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace knlp {

EmbeddingOptions::EmbeddingOptions(int64_t count, int64_t dimension)
    : count_(count), dimension_(dimension) {}

EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options) : options(options) {
  reset();
}

void EmbeddingImpl::reset() {
  weight = register_parameter(
      "weight", torch::empty({options.count_, options.dimension_}));
  torch::NoGradGuard guard;
  weight.normal_(0, options.init_max_value_);
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(count=" << options.count_
         << ", dimension=" << options.dimension_ << ")";
}

Tensor EmbeddingImpl::forward(const Tensor& input) {
  return torch::embedding(weight, /*indices=*/input);
}
}  // namespace knlp
