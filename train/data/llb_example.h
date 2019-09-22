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

#include "torch/torch.h"
namespace knlp {
namespace data {
using Tensor = torch::Tensor;
class LlbExample {
 public:
  virtual ~LlbExample(){};
  std::vector<Tensor> features;
  Tensor target;
};

}  // namespace data
}  // namespace knlp
