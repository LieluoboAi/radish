/*
 * File: llb_model.h
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-19 6:13:08
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <torch/torch.h>
#include <memory>

namespace knlp {
namespace train {
using Tensor = torch::Tensor;
class TORCH_API LlbModel : public ::torch::nn::Module {
 public:
  virtual ~LlbModel() {}

  /**
   *
   * 有监督时有target，无监督时target没有设置
   * logits  ->  模型输出
   * target    -> 标注标签
   **/
  virtual Tensor CalcLoss(const std::vector<Tensor>& examples, const Tensor& logits, const Tensor& target = {}) = 0;

  /**
   *
   * 评估模型，在测试数据上，返回值类似loss, 越低表示模型效果越好
   * */
  virtual Tensor EvalModel(const std::vector<Tensor>& examples, const Tensor& loggit, const Tensor& target = {}) = 0;

  virtual Tensor forward(std::vector<Tensor> inputs) = 0;
};

}  // namespace train
}  // namespace knlp
