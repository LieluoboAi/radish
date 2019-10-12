/*
 * File: positionwise_fc.h
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 3:14:58
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <cstddef>
#include <vector>

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/container/any.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = ::torch::Tensor;
/// Options for the `ScaleProductAttention` module.
struct TORCH_API PositionwiseFCOptions {
  PositionwiseFCOptions(int64_t d_in, int64_t d_hidden, double dropout);
  TORCH_ARG(int64_t, d_in);
  TORCH_ARG(int64_t, d_hidden);
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API PositionwiseFCImpl
    : public torch::nn::Cloneable<PositionwiseFCImpl> {
 public:
  PositionwiseFCImpl(int64_t d_in, int64_t d_hidden, double dropout)
      : PositionwiseFCImpl(PositionwiseFCOptions(d_in, d_hidden, dropout)) {}
  explicit PositionwiseFCImpl(PositionwiseFCOptions options);

  void reset() override;

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The `Options` used to configure this  module.
  PositionwiseFCOptions options;
  torch::nn::AnyModule in2hidden;
  torch::nn::AnyModule hidden2in;
  torch::nn::AnyModule dropout;
  torch::nn::AnyModule layernorm;
};

/// A `ModuleHolder` subclass for `PositionwiseFCImpl`.
/// See the documentation for `PositionwiseFCImpl` class to learn what methods
/// it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(PositionwiseFC);
}  // namespace radish
