/*
 * File: llb_example.h
 * Project: data
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-19 5:59:22
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <torch/torch.h>
#include "train/proto/example.pb.h"

namespace knlp {
namespace data {
using Tensor = torch::Tensor;
class LlbExample {
 public:
  virtual ~LlbExample(){};
  // 把特征转化为feature list, 每个特征/组一个tensor
  virtual std::vector<Tensor> ToTensorList() = 0;
  virtual bool FromMessage(const train::TrainExample& ex) = 0;
  Tensor target;
};

}  // namespace data
}  // namespace knlp
