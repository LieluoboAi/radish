/*
 * File: span_bert_model.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-20 9:38:26
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "bert/span_bert_model.h"

namespace knlp {
SpanBertOptions::SpanBertOptions(int64_t n_src_vocab, int64_t len_max_seq,
                                 int64_t d_word_vec, int64_t n_layers,
                                 int64_t n_head, int64_t d_k,
                                 int64_t d_v, int64_t d_model,
                                 int64_t d_inner, double dropout)
    : n_src_vocab_(n_src_vocab),
      len_max_seq_(len_max_seq),
      d_word_vec_(d_word_vec),
      n_layers_(n_layers),
      n_head_(n_head),
      d_k_(d_k),
      d_v_(d_v),
      d_model_(d_model),
      d_inner_(d_inner),
      dropout_(dropout) {}
SpanBertModelImpl::SpanBertModelImpl(SpanBertOptions options_)
    : options(options_) {
  encoder = TransformerEncoder(
      options.n_src_vocab_, options.len_max_seq_, options.d_word_vec_,
      options.n_layers_, options.n_head_, options.d_k_, options.d_k_,
      options.d_model_, options.d_inner_, options.dropout_);
  register_module("transformer_encoder", encoder);
}
Tensor SpanBertModelImpl::CalcLoss(const std::vector<Tensor>& examples,
                                   const Tensor& logits, const Tensor& target) {
  return {};
}

/**
 *
 * 评估模型，在测试数据上，返回值类似loss, 越低表示模型效果越好
 * */
Tensor SpanBertModelImpl::EvalModel(const std::vector<Tensor>& examples,
                                    const Tensor& loggit,
                                    const Tensor& target) {
  return {};
}

Tensor SpanBertModelImpl::forward(std::vector<Tensor> inputs) { return {}; }

}  // namespace knlp
