/*
 * File: query_same_model.cc
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-05 6:23:08
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "bert/finetune/query_same_model.h"
#include "utils/logging.h"

namespace radish {
static Tensor batch_select(const Tensor& input, const Tensor& inds) {
  // input [B,N,D]
  // inds [B,S]  -> [B,S,D]
  // ==>output: [B,S,D]
  Tensor dummy =
      inds.unsqueeze(2).expand({inds.size(0), inds.size(1), input.size(2)});
  return input.gather(1, dummy);
}

static Tensor calc_loss_(const Tensor& pred_, const Tensor& target) {
  Tensor gold = target.contiguous().view(-1);
  int dim = pred_.dim();
  int firstDimSize = pred_.size(0);
  if (dim == 3) {
    firstDimSize = pred_.size(0) * pred_.size(1);
  }
  Tensor pred = pred_.view({firstDimSize, -1});
  Tensor one_hot = torch::zeros_like(pred).scatter_(1, gold.view({-1, 1}), 1);
  Tensor log_prb = torch::log_softmax(pred, 1);
  Tensor xent = one_hot.mul_(log_prb).neg_().sum(1);
  return xent.mean();
}

static Tensor calc_accuracy_(const Tensor& pred, const Tensor& target) {
  torch::NoGradGuard guard;
  int dim = pred.dim() - 1;
  Tensor predT = pred.argmax(dim).contiguous().view(-1);
  Tensor gold = target.contiguous().view(-1);
  Tensor correct = predT.eq(gold);
  return correct.toType(torch::kFloat32).mean();
}
QuerySameOptions::QuerySameOptions(int64_t n_src_vocab)
    : n_src_vocab_(n_src_vocab) {}

QuerySameModelImpl::QuerySameModelImpl(QuerySameOptions options_)
    : options(options_) {
  encoder = TransformerEncoder(
      TransformerEncoderOptions(
          options.n_src_vocab_, options.len_max_seq_, options.d_word_vec_,
          options.n_layers_, options.n_head_, options.d_k_, options.d_v_,
          options.d_model_, options.d_inner_, options.dropout_)
          .need_factor_embedding(true));
  register_module("transformer_encoder", encoder);
  final_proj = torch::nn::Linear(options.d_model_, 2);
  register_module("final_proj", final_proj);
  torch::NoGradGuard guard;
  torch::nn::init::xavier_normal_(final_proj->weight);
}

Tensor QuerySameModelImpl::CalcLoss(const std::vector<Tensor>& inputs,
                                    const Tensor& logits,
                                    std::vector<float>& evals,
                                    const Tensor& target, bool train) {
  Tensor firstTokenRepr = logits.select(1, 0);
  Tensor preds = final_proj(firstTokenRepr);
  Tensor loss = calc_loss_(preds, target);
  if (!train) {
    float accuracy = calc_accuracy_(preds, target).item().to<float>();
    evals.push_back(accuracy);
  }
  return loss;
}

/**
 *inputs- 0 -src_seq
 *        1 - types
 *
 */
Tensor QuerySameModelImpl::forward(std::vector<Tensor> inputs) {
  CHECK(inputs.size() >= 2);
  // 0 - for seq
  Tensor& src_seq = inputs[0];
  auto seqLen = src_seq.size(1);
  Tensor pos_seq = torch::arange(
      0, seqLen,
      torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
  // should be same device as src seq
  pos_seq = pos_seq.repeat({src_seq.size(0), 1}).to(src_seq.device());

  // types
  Tensor& types = inputs[1];
  auto rets = encoder(src_seq, pos_seq, types);
  return rets[0];
}

}  // namespace radish
