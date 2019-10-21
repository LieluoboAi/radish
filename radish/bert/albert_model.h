
#pragma once

#include "radish/layers/layer_norm.h"
#include "radish/train/llb_model.h"
#include "radish/bert/model/bert_model.h"
namespace radish {
using Tensor = torch::Tensor;

class TORCH_API ALBertModelImpl : public train::LlbModel {
 public:
  explicit ALBertModelImpl(BertOptions options);

  Tensor CalcLoss(const std::vector<Tensor> &examples,
                                      const std::vector<Tensor> &logits,
                                      std::vector<float>& evals,
                                      const Tensor &target = {}) override;

  std::vector<Tensor> forward(std::vector<Tensor> inputs) override;

  BertOptions options;
  BertModel bert = nullptr;
  LayerNorm laynorm = nullptr;
  torch::nn::Linear vocab_proj = nullptr;
  torch::nn::Linear order_proj = nullptr;
};

TORCH_MODULE(ALBertModel);
}  // namespace radish
