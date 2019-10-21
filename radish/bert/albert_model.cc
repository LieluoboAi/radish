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
#include "radish/bert/albert_model.h"
#include "radish/utils/logging.h"

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

ALBertModelImpl::ALBertModelImpl(BertOptions options_) : options(options_) {
  bert = BertModel(options);
  register_module("bert", bert);
  vocab_proj = torch::nn::Linear(options.d_wordvec(), options.n_vocab());
  register_module("vocab_proj", vocab_proj);
  order_proj = torch::nn::Linear(options.hidden_size(), 2);
  register_module("order_proj", order_proj);
  laynorm = LayerNorm(options.d_wordvec(), options.ln_eps());
  register_module("laynorm", laynorm);
  torch::NoGradGuard guard;
  vocab_proj->weight = bert->embeddings->word_embeddings->weight;
  torch::nn::init::normal_(order_proj->weight, 0, options.init_range());
  torch::nn::init::constant_(vocab_proj->bias, 0);
  torch::nn::init::constant_(order_proj->bias, 0);
}

Tensor ALBertModelImpl::CalcLoss(const std::vector<Tensor>& inputs,
                                 const std::vector<Tensor>& logits,
                                 std::vector<float>& evals,
                                 const Tensor& target) {
  Tensor mlm_loss = calc_loss_(logits[0], target, true);
  if (!is_training()) {
    float mlm_accuracy =
        calc_accuracy_(logits[0], target, true).item().to<float>();
    evals.push_back(mlm_accuracy);
  }

  //  inputs[3] is the ordered target
  Tensor order_loss = calc_loss_(logits[1], inputs[3], false);
  if (!is_training()) {
    float order_accuracy =
        calc_accuracy_(logits[1], inputs[3], false).item().to<float>();
    evals.push_back(order_accuracy);
  }
  return mlm_loss.add(order_loss);
}

/**
 *inputs- 0 -src_seq
 *        1 - masked_indexies
 *        2 - types
 *        3- ordered
 *
 */
std::vector<Tensor> ALBertModelImpl::forward(std::vector<Tensor> inputs) {
  CHECK(inputs.size() >= 4);
  // 0 - for seq
  Tensor& src_seq = inputs[0];
  Tensor mask = src_seq.ne(0).toType(torch::kFloat32).to(src_seq.device());
  // types
  Tensor& types = inputs[2];
  auto rets = bert(src_seq, mask, types);

  Tensor maskedOutput = batch_select(rets[0], inputs[1]);
  int bsz = maskedOutput.size(0);
  int numTargets = maskedOutput.size(1);
  int hidden = maskedOutput.size(2);
  Tensor maskPreds = maskedOutput.view({-1, hidden})
                         .mm(bert->embeddings->embedding_to_hidden_proj->weight)
                         .view({bsz, numTargets, -1});
  maskPreds = laynorm(maskPreds);
  maskPreds = vocab_proj(maskPreds);

  Tensor orderPreds = order_proj(rets[1]);
  return {maskPreds, orderPreds};
}

}  // namespace radish
