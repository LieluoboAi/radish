/*
 * File: scale_product_attention.h
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 11:25:43
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/nn/modules/any.h>
#include <cstddef>
#include <vector>

namespace knlp {
using Tensor = ::torch::Tensor;
/// Options for the `ScaleProductAttention` module.
struct TORCH_API ScaleProductAttentionOptions {
  ScaleProductAttentionOptions(double temperature, double att_dropout);
  TORCH_ARG(double, temperature)=1.0;
  TORCH_ARG(double, att_dropout) = 0.1;
};

class TORCH_API ScaleProductAttentionImpl : public torch::nn::Cloneable<ScaleProductAttentionImpl> {
 public:
  ScaleProductAttentionImpl(double temperature, double att_dropout=0.1)
      : ScaleProductAttentionImpl(ScaleProductAttentionOptions(temperature, att_dropout)) {}
  explicit ScaleProductAttentionImpl(ScaleProductAttentionOptions options);

  void reset() override;

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  std::vector<Tensor> forward(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask={});

  /// The `Options` used to configure this  module.
  ScaleProductAttentionOptions options;

  torch::nn::AnyModule dropout;
};

/// A `ModuleHolder` subclass for `ScaleProductAttentionImpl`.
/// See the documentation for `ScaleProductAttentionImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ScaleProductAttention);

}  // namespace knlp
