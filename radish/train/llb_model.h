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
  virtual Tensor CalcLoss(const std::vector<Tensor>& examples,
                          const std::vector<Tensor>& logits,
                          std::vector<float>& evals,
                          const Tensor& target = {}) = 0;
  virtual std::vector<Tensor> forward(std::vector<Tensor> inputs) = 0;

  virtual bool LoadFromPretrain(std::string path) { return false; }

  /**
   * 如果返回true,
   *那么将使用batch模式在测试集上eval，最后各项指标会按batch数求平均
   * 反之，会在整个测试集上进行,一次eval完
   *
   **/
  virtual bool EvalInBatch() const { return false; }
};

}  // namespace train
}  // namespace radish
