/*
 * File: query_same_model.h
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-03 6:20:34
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include "radish/train/llb_model.h"
#include "radish/transformer/transformer_encoder.h"
namespace radish {
using Tensor = torch::Tensor;

/// Options for the `QuerySame` module.
struct TORCH_API QuerySameOptions {
  QuerySameOptions(int64_t n_src_vocab);
  TORCH_ARG(int64_t, n_src_vocab)=32003;
  TORCH_ARG(int64_t, len_max_seq)=521;
  TORCH_ARG(int64_t, d_word_vec)=128;
  TORCH_ARG(int64_t, n_layers)=5;
  TORCH_ARG(int64_t, n_head)=8;
  TORCH_ARG(int64_t, d_k)=40;
  TORCH_ARG(int64_t, d_v)=40;
  TORCH_ARG(int64_t, d_model)=320;
  TORCH_ARG(int64_t, d_inner)=1280;
  TORCH_ARG(int64_t, n_class)=2; // for XNLI, there are 3 classes
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API QuerySameModelImpl : public train::LlbModel {
 public:
  explicit QuerySameModelImpl(int64_t n_src_vocab)
      : QuerySameModelImpl(
            QuerySameOptions(n_src_vocab)) {}
  explicit QuerySameModelImpl(QuerySameOptions options);

  Tensor CalcLoss(const std::vector<Tensor> &examples,
                                      const Tensor &logits,
                                      std::vector<float>& evals,
                                      const Tensor &target = {},
                                      bool train = true) override;

  Tensor forward(std::vector<Tensor> inputs) override;

  QuerySameOptions options;
  TransformerEncoder encoder = nullptr;
  torch::nn::Linear final_proj = nullptr;
};

TORCH_MODULE(QuerySameModel);
}  // namespace radish
