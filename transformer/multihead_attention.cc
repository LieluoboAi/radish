/*
 * File: multihead_attention.cc
 * Project: transformer
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-16 4:19:36
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "transformer/multihead_attention.h"

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "torch/nn/init.h"
#include "torch/nn/modules/dropout.h"
#include "torch/types.h"
#include "torch/utils.h"

#include "layers/layer_norm.h"
#include "utils/logging.h"

namespace radish {

MultiheadAttentionOptions::MultiheadAttentionOptions(int64_t n_head,
                                                     int64_t d_model,
                                                     int64_t d_k, int64_t d_v,
                                                     double dropout)
    : n_head_(n_head),
      d_model_(d_model),
      d_k_(d_k),
      d_v_(d_v),
      dropout_(dropout) {}

MultiheadAttentionImpl::MultiheadAttentionImpl(
    MultiheadAttentionOptions options_)
    : options(options_) {
  reset();
}

void MultiheadAttentionImpl::reset() {
  w_qs = torch::nn::Linear(options.d_model_, options.n_head_ * options.d_k_);
  register_module("w_qs", w_qs);
  w_ks = torch::nn::Linear(options.d_model_, options.n_head_ * options.d_k_);
  register_module("w_ks", w_ks);
  w_vs = torch::nn::Linear(options.d_model_, options.n_head_ * options.d_v_);
  register_module("w_vs", w_vs);
  attention =
      radish::ScaleProductAttention(std::sqrt(options.d_k_), options.dropout_);
  register_module("attention", attention);
  fc = torch::nn::Linear(options.n_head_ * options.d_v_, options.d_model_);
  register_module("fc", fc);
  layernorm = register_module("layernorm", radish::LayerNorm(options.d_model_));
  dropout = register_module("dropout", torch::nn::Dropout(options.dropout_));

  torch::NoGradGuard guard;
  torch::nn::init::normal_(
      w_qs->weight, 0,
      std::sqrt(2.0 / (options.d_model_ + options.d_k_ + 0.000001)));
  torch::nn::init::constant_(w_qs->bias, 0);
  torch::nn::init::normal_(
      w_ks->weight, 0,
      std::sqrt(2.0 / (options.d_model_ + options.d_k_ + 0.000001)));
  torch::nn::init::constant_(w_ks->bias, 0);
  torch::nn::init::normal_(
      w_vs->weight, 0,
      std::sqrt(2.0 / (options.d_model_ + options.d_v_ + 0.000001)));
  torch::nn::init::constant_(w_vs->bias, 0);
  torch::nn::init::xavier_normal_(fc->weight);
  torch::nn::init::constant_(fc->bias, 0);
}

void MultiheadAttentionImpl::pretty_print(std::ostream& stream) const {
  stream << "transformer::MultiheadAttention(dropout=" << options.dropout_
         << ", n_head=" << options.n_head_ << ", d_model=" << options.d_model_
         << ", d_k=" << options.d_k_ << ", d_v=" << options.d_v_ << ")";
}

std::vector<Tensor> MultiheadAttentionImpl::forward(const Tensor& q,
                                                    const Tensor& k,
                                                    const Tensor& v,
                                                    const Tensor& mask) {
  CHECK_EQ(q.sizes(), k.sizes());
  CHECK_EQ(q.sizes(), v.sizes());
  CHECK_EQ(q.ndimension(), 3);

  int64_t n_head = options.n_head_;
  int64_t d_k = options.d_k_;
  int64_t d_v = options.d_v_;

  int64_t len_q = q.size(1);
  int64_t len_k = k.size(1);
  int64_t len_v = v.size(1);
  int64_t sz_b = q.size(0);

  const auto residual = q;
  auto q_ = w_qs->forward(q).view({sz_b, len_q, n_head, d_k});
  auto k_ = w_ks->forward(k).view({sz_b, len_k, n_head, d_k});
  auto v_ = w_vs->forward(v).view({sz_b, len_v, n_head, d_v});

  q_ = q_.permute({2, 0, 1, 3})
           .contiguous()
           .view({-1, len_q, d_k});  // (n*b) x lq x dk
  k_ = k_.permute({2, 0, 1, 3})
           .contiguous()
           .view({-1, len_k, d_k});  // (n*b) x lk x dk
  v_ = v_.permute({2, 0, 1, 3})
           .contiguous()
           .view({-1, len_v, d_v});  // (n*b) x lv x dv

  auto mask_ = mask.repeat({n_head, 1, 1});  // (n*b) x .. x ..
  std::vector<Tensor> rets = attention(q_, k_, v_, mask_);
  auto& output = rets[0];
  const auto& attn = rets[1];
  output = output.view({n_head, sz_b, len_q, d_v});
  output = output.permute({1, 2, 0, 3})
               .contiguous()
               .view({sz_b, len_q, -1});  //  b x lq x (n*dv)

  output = dropout.forward(fc->forward(output));
  output.add_(residual);
  output = layernorm.forward(output);
  return {output, attn};
}
}  // namespace radish
