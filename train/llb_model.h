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
#include <tuple>

namespace radish {
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
   * return   first tensor for loss, second for eval
   *
   **/
  virtual std::tuple<Tensor, Tensor> CalcLoss(
      const std::vector<Tensor>& examples, const Tensor& logits,
      const Tensor& target = {}, bool train=true) = 0;

  virtual Tensor forward(std::vector<Tensor> inputs) = 0;
};

}  // namespace train
}  // namespace radish
