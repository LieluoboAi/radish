/*
 * File: span_bert_model.h
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-19 8:06:07
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include "train/llb_model.h"
#include "transformer/transformer_encoder.h"
namespace radish {
using Tensor = torch::Tensor;

/// Options for the `SpanBert` module.
struct TORCH_API SpanBertOptions {
  SpanBertOptions(int64_t n_src_vocab, int64_t len_max_seq, int64_t d_word_vec,
                  int64_t n_layers = 6, int64_t n_head = 8, int64_t d_k = 25,
                  int64_t d_v = 25, int64_t d_model = 200,
                  int64_t d_inner = 200, double dropout = 0.1);
  TORCH_ARG(int64_t, n_src_vocab);
  TORCH_ARG(int64_t, len_max_seq);
  TORCH_ARG(int64_t, d_word_vec);
  TORCH_ARG(int64_t, n_layers);
  TORCH_ARG(int64_t, n_head);
  TORCH_ARG(int64_t, d_k);
  TORCH_ARG(int64_t, d_v);
  TORCH_ARG(int64_t, d_model);
  TORCH_ARG(int64_t, d_inner);
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API SpanBertModelImpl : public train::LlbModel {
 public:
  SpanBertModelImpl(int64_t n_src_vocab, int64_t len_max_seq,
                    int64_t d_word_vec)
      : SpanBertModelImpl(
            SpanBertOptions(n_src_vocab, len_max_seq, d_word_vec)) {}
  explicit SpanBertModelImpl(SpanBertOptions options);

  std::tuple<Tensor, Tensor> CalcLoss(const std::vector<Tensor>& examples,
                                      const Tensor& logits,
                                      const Tensor& target = {}) override;

  Tensor forward(std::vector<Tensor> inputs) override;

  SpanBertOptions options;
  TransformerEncoder encoder = nullptr;
  torch::nn::Linear proj = nullptr;
  torch::nn::Linear span_proj = nullptr;
};

TORCH_MODULE(SpanBertModel);
}  // namespace radish
