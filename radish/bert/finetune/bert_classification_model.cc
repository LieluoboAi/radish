/*
 * File: bert_classification_model.cc
 * Project: finetune
 * File Created: Sunday, 20th October 2019 9:09:08 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 9:09:11 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */
#include "radish/bert/finetune/bert_classification_model.h"
#include "radish/train/model_io.h"
#include "radish/utils/logging.h"

namespace radish {

bool BertClassificationModelImpl::LoadFromPretrain(std::string path) {
  ::radish::train::LoadModel(bert.ptr(), path, "", torch::kCPU, true);
  return true;
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

BertClassificationModelImpl::BertClassificationModelImpl(BertOptions options_,
                                                         int ncls)
    : options(options_), n_class(ncls) {
  bert = BertModel(options);
  register_module("bert", bert);
  final_proj = torch::nn::Linear(options.hidden_size(), n_class);
  register_module("final_proj", final_proj);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(final_proj->weight, 0, options.init_range());
  torch::nn::init::constant_(final_proj->bias, 0);
}

static void debug_(std::string title, const Tensor& t) {
  auto maxv = t.max().item().to<float>();
  auto minv = t.min().item().to<float>();
  auto meanv = t.mean().item().to<float>();
  spdlog::info("{}  max={},min={}, mean={}", title, maxv, minv, meanv);
}
Tensor BertClassificationModelImpl::CalcLoss(const std::vector<Tensor>& inputs,
                                             const Tensor& logits,
                                             std::vector<float>& evals,
                                             const Tensor& target, bool train) {
  Tensor loss = calc_loss_(logits, target);
  if (!train) {
    float accuracy = calc_accuracy_(logits, target).item().to<float>();
    evals.push_back(accuracy);
  }
  return loss;
}

/**
 *inputs- 0 -src_seq
 *        1 - types
 *
 */
Tensor BertClassificationModelImpl::forward(std::vector<Tensor> inputs) {
  CHECK(inputs.size() >= 2);
  // 0 - for seq
  Tensor& src_seq = inputs[0];
  Tensor mask = src_seq.ne(0).toType(torch::kFloat32).to(src_seq.device());
  // types
  Tensor& types = inputs[1];
  auto rets = bert(src_seq, mask, types);
  Tensor hiddens = rets[1];  // pooled
  return final_proj(hiddens);
}

}  // namespace radish
