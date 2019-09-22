/*
 * File: encoder_layer.h
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 6:14:37
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

#include "transformer/multihead_attention.h"

namespace radish {
using Tensor = torch::Tensor;
/// Options for the `EncoderLayer` module.
struct TORCH_API EncoderLayerOptions {
  EncoderLayerOptions(int64_t d_model, int64_t d_inner, int64_t n_head,
                      int64_t d_k, int64_t d_v, double dropout = 0.1);
  TORCH_ARG(int64_t, d_model);
  TORCH_ARG(int64_t, d_inner);
  TORCH_ARG(int64_t, n_head);
  TORCH_ARG(int64_t, d_k);
  TORCH_ARG(int64_t, d_v);
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API EncoderLayerImpl
    : public ::torch::nn::Cloneable<EncoderLayerImpl> {
 public:
  EncoderLayerImpl(int64_t d_model, int64_t d_inner, int64_t n_head,
                   int64_t d_k, int64_t d_v, double dropout = 0.1)
      : EncoderLayerImpl(
            EncoderLayerOptions(d_model, d_inner, n_head, d_k, d_v, dropout)) {}
  explicit EncoderLayerImpl(EncoderLayerOptions options);

  void reset() override;

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  std::vector<Tensor> forward(const Tensor& enc_input,
                              const Tensor& non_pad_mask = {},
                              const Tensor& slf_attn_mask = {});

  /// The options used to configure this module.
  EncoderLayerOptions options;
  radish::MultiheadAttention slf_attn = nullptr;
  torch::nn::AnyModule pos_ffn;
};

/// A `ModuleHolder` subclass for `EncoderLayerImpl`.
/// See the documentation for `EncoderLayerImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(EncoderLayer);

}  // namespace radish
