/*
 * File: transformer.h
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-14 10:08:03
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>

#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

#include "transformer/transformer_decoder.h"
#include "transformer/transformer_encoder.h"

namespace knlp {
using Tensor = torch::Tensor;
/// Options for the `Transformer` module.
struct TORCH_API TransformerOptions {
  TransformerOptions(int64_t n_src_vocab, int64_t n_tgt_vocab,
                     int64_t len_max_seq, int64_t d_word_vec = 200,
                     int64_t d_model = 200, int64_t d_inner = 800,
                     int64_t n_layers = 6, int64_t n_head = 8, int64_t d_k = 25,
                     int64_t d_v = 25, double dropout = 0.1,
                     bool tgt_emb_prj_weight_sharing = true,
                     bool emb_src_tgt_weight_sharing = true);
  TORCH_ARG(int64_t, n_src_vocab);
  TORCH_ARG(int64_t, n_tgt_vocab);
  TORCH_ARG(int64_t, len_max_seq);
  TORCH_ARG(int64_t, d_word_vec) = 200;
  TORCH_ARG(int64_t, d_model) = 200;
  TORCH_ARG(int64_t, d_inner) = 800;
  TORCH_ARG(int64_t, n_layers) = 6;
  TORCH_ARG(int64_t, n_head) = 8;
  TORCH_ARG(int64_t, d_k) = 25;
  TORCH_ARG(int64_t, d_v) = 25;
  TORCH_ARG(double, dropout) = 0.1;
  TORCH_ARG(bool, tgt_emb_prj_weight_sharing) = true;
  TORCH_ARG(bool, emb_src_tgt_weight_sharing) = true;
};

class TORCH_API TransformerImpl
    : public ::torch::nn::Cloneable<TransformerImpl> {
 public:
  TransformerImpl(int64_t n_src_vocab, int64_t n_tgt_vocab, int64_t len_max_seq)
      : TransformerImpl(
            TransformerOptions(n_src_vocab, n_tgt_vocab, len_max_seq)) {}
  explicit TransformerImpl(TransformerOptions options);

  void reset() override;

  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& src_seq, const Tensor& src_pos,
                              const Tensor& tgt_seq, const Tensor& tgt_pos);

  /// The options used to configure this module.
  TransformerOptions options;
  torch::nn::Linear tgt_word_prj = nullptr;
  TransformerEncoder encoder = nullptr;
  TransformerDecoder decoder = nullptr;
  double x_logit_scale = 1.0;
};

/// A `ModuleHolder` subclass for `TransformerImpl`.
/// See the documentation for `TransformerImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(Transformer);

}  // namespace knlp
