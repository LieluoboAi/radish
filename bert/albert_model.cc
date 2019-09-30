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
#include "bert/albert_model.h"
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

static Tensor calc_loss_(const Tensor& pred_, const Tensor& target,
                         bool labelSmoothing) {
  Tensor gold = target.contiguous().view(-1);
  int dim = pred_.dim();
  int firstDimSize = pred_.size(0);
  if (dim == 3) {
    firstDimSize = pred_.size(0) * pred_.size(1);
  }
  Tensor pred = pred_.view({firstDimSize, -1});
  int n_class = pred.size(1);
  Tensor one_hot = torch::zeros_like(pred).scatter_(1, gold.view({-1, 1}), 1);
  float normalizing = 0;
  if (labelSmoothing) {
    float lowp = 0.1 / static_cast<float>(n_class - 1);
    normalizing =
        -(0.9 * log(0.9) + float(n_class - 1) * lowp * log(lowp + 1e-20));
    one_hot.mul_(0.9 - lowp).add_(lowp);
  }
  Tensor log_prb = torch::log_softmax(pred, 1);
  Tensor xent = one_hot.mul_(log_prb).neg_().sum(1);
  if (labelSmoothing) {
    Tensor non_pad_mask = gold.ne(0);
    xent = xent.masked_select(non_pad_mask);
  }
  return labelSmoothing ? xent.sub_(normalizing).mean() : xent.mean();
}

static Tensor calc_accuracy_(const Tensor& pred, const Tensor& target,
                             bool mask) {
  torch::NoGradGuard guard;
  int dim = pred.dim() - 1;
  Tensor predT = pred.argmax(dim).contiguous().view(-1);
  Tensor gold = target.contiguous().view(-1);
  Tensor correct = predT.eq(gold);
  if (mask) {
    Tensor non_pad_mask = gold.ne(0);
    correct = correct.masked_select(non_pad_mask);
  }
  return correct.toType(torch::kFloat32).mean();
}
ALBertOptions::ALBertOptions(int64_t n_src_vocab) : n_src_vocab_(n_src_vocab) {}

ALBertModelImpl::ALBertModelImpl(ALBertOptions options_) : options(options_) {
  encoder = TransformerEncoder(
      TransformerEncoderOptions(
          options.n_src_vocab_, options.len_max_seq_, options.d_word_vec_,
          options.n_layers_, options.n_head_, options.d_k_, options.d_v_,
          options.d_model_, options.d_inner_, options.dropout_)
          .need_factor_embedding(true));
  register_module("transformer_encoder", encoder);
  vocab_proj = torch::nn::Linear(options.d_word_vec_, options.n_src_vocab_);
  register_module("vocab_proj", vocab_proj);
  order_proj = torch::nn::Linear(options.d_model_, 2);
  register_module("order_proj", order_proj);
  laynorm = LayerNorm(options.d_word_vec_);
  register_module("laynorm", laynorm);
  torch::NoGradGuard guard;
  vocab_proj->weight = encoder->src_word_emb->weight;
  torch::nn::init::xavier_normal_(order_proj->weight);
}

Tensor ALBertModelImpl::CalcLoss(const std::vector<Tensor>& inputs,
                                 const Tensor& logits,
                                 std::vector<float>& evals,
                                 const Tensor& target, bool train) {
  Tensor maskedOutput = batch_select(logits, inputs[1]);
  int bsz = maskedOutput.size(0);
  int numTargets = maskedOutput.size(1);
  int hidden = maskedOutput.size(2);
  Tensor maskPreds = maskedOutput.view({-1, hidden})
                         .mm(encoder->embedding_to_hidden_proj->weight)
                         .view({bsz, numTargets, -1});
  maskPreds = laynorm(maskPreds);
  maskPreds = vocab_proj(maskPreds);
  Tensor mlm_loss = calc_loss_(maskPreds, target, true);
  if (!train) {
    float mlm_accuracy =
        calc_accuracy_(maskPreds, target, true).item().to<float>();
    evals.push_back(mlm_accuracy);
  }

  Tensor firstTokenRepr = logits.select(1, 0);
  Tensor orderPreds = order_proj(firstTokenRepr);
  //  inputs[3] is the ordered target
  Tensor order_loss = calc_loss_(orderPreds, inputs[3], false);
  if (!train) {
    float order_accuracy =
        calc_accuracy_(orderPreds, inputs[3], false).item().to<float>();
    evals.push_back(order_accuracy);
  }
  return mlm_loss.add_(order_loss);
}

/**
 *inputs- 0 -src_seq
 *        1 - masked_indexies
 *        2 - types
 *        3- ordered
 *
 */
Tensor ALBertModelImpl::forward(std::vector<Tensor> inputs) {
  CHECK(inputs.size() >= 4);
  // 0 - for seq
  Tensor& src_seq = inputs[0];
  auto seqLen = src_seq.size(1);
  Tensor pos_seq = torch::arange(
      0, seqLen,
      torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
  // should be same device as src seq
  pos_seq = pos_seq.repeat({src_seq.size(0), 1}).to(src_seq.device());

  // types
  Tensor& types = inputs[2];
  auto rets = encoder(src_seq, pos_seq, types);
  return rets[0];
}

}  // namespace radish
