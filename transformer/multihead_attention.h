/*
 * File: multihead_attention.h
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-14 10:17:10
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

#include "transformer/scale_product_attention.h"

namespace radish {
using Tensor = torch::Tensor;
/// Options for the `MultiheadAttention` module.
struct TORCH_API MultiheadAttentionOptions {
  MultiheadAttentionOptions(int64_t n_head, int64_t d_model, int64_t d_k,
                            int64_t d_v, double dropout = 0.1);
  TORCH_ARG(int64_t, n_head);
  TORCH_ARG(int64_t, d_model);
  TORCH_ARG(int64_t, d_k);
  TORCH_ARG(int64_t, d_v);
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API MultiheadAttentionImpl
    : public ::torch::nn::Cloneable<MultiheadAttentionImpl> {
 public:
  MultiheadAttentionImpl(int64_t n_head, int64_t d_model, int64_t d_k,
                         int64_t d_v, double dropout = 0.1)
      : MultiheadAttentionImpl(
            MultiheadAttentionOptions(n_head, d_model, d_k, d_v, dropout)) {}
  explicit MultiheadAttentionImpl(MultiheadAttentionOptions options);

  void reset() override;

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  std::vector<Tensor> forward(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& mask = {});

  /// The options used to configure this module.
  MultiheadAttentionOptions options;
  torch::nn::Linear w_qs = nullptr;
  torch::nn::Linear w_ks = nullptr;
  torch::nn::Linear w_vs = nullptr;
  radish::ScaleProductAttention attention = nullptr;
  torch::nn::AnyModule layernorm;
  torch::nn::Linear fc = nullptr;
  torch::nn::AnyModule dropout;
};

/// A `ModuleHolder` subclass for `MultiheadAttentionImpl`.
/// See the documentation for `MultiheadAttentionImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(MultiheadAttention);

}  // namespace radish
