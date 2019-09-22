/*
 * File: layer_norm.h
 * Project: layers
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-15 6:08:56
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <cstddef>
#include <vector>

namespace radish {
using Tensor = ::torch::Tensor;
/// Options for the `LayerNorm` module.
struct TORCH_API LayerNormOptions {
  LayerNormOptions(torch::IntArrayRef shape, double eps = 0.00001,
                   bool elementAffine = true);
  TORCH_ARG(torch::IntArrayRef, shape);
  // eps
  TORCH_ARG(double, eps) = 0.00001;
  // element affine
  TORCH_ARG(bool, element_affine) = true;
};

class TORCH_API LayerNormImpl : public torch::nn::Cloneable<LayerNormImpl> {
 public:
  LayerNormImpl(torch::IntArrayRef shape, double eps)
      : LayerNormImpl(LayerNormOptions(shape, eps, true)) {}
  explicit LayerNormImpl(LayerNormOptions options);

  void reset() override;

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The `Options` used to configure this `v` module.
  LayerNormOptions options;

  Tensor weight;
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LayerNormImpl`.
/// See the documentation for `LayerNormImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(LayerNorm);

}  // namespace radish
