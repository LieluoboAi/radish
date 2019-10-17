/*
 * File: bert_options.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 10:03:58
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include "torch/torch.h"

/// Options for the Bert relates .
struct TORCH_API BertOptions {
  BertOptions(int64_t n_vocab, int64_t hidden_size)
      : n_vocab_(n_vocab), hidden_size_(hidden_size){};
  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, n_vocab);
  // embedding size
  TORCH_ARG(int64_t, hidden_size)=200;
  // intermediate  size
  TORCH_ARG(int64_t, intermediate_size)=800;

  // max pos
  TORCH_ARG(int64_t, max_pos)=512;

  // max types
  TORCH_ARG(int64_t, max_types)=3;

  TORCH_ARG(int64_t, num_heads)=8;
  TORCH_ARG(int64_t, num_layers)=5;

  TORCH_ARG(bool, output_attentions)=false;

  TORCH_ARG(double, ln_eps) = 1e-12;
  TORCH_ARG(double, init_range) = 0.02;
  TORCH_ARG(double, dropout) = 0.1;
};
