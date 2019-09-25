/*
 * File: tensor_util.cc
 * Project: utils
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-25 10:23:38
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "utils/tensor_util.h"

namespace radish {
namespace utils {

bool IsEmpty(const torch::Tensor& x) {
  if (x.defined() && x.dim() > 0 && x.size(0) != 0 && x.numel() > 0) {
    return false;
  } else {
    return true;
  }
}

void ClipGradienNorm(std::vector<at::Tensor>& parameters, float max_norm) {
  double total_norm = 0.0;
  for (auto& p : parameters) {
    if (p.requires_grad()) {
      auto param_norm = p.grad();
      if (!IsEmpty(param_norm)) {
        param_norm = param_norm.norm();
        total_norm += std::pow(param_norm.item<float>(), 2.f);
      }
    }
  }
  total_norm = std::pow(total_norm, (1. / 2.));
  auto clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1) {
    for (at::Tensor& p : parameters) {
      auto param_norm = p.grad();
      if (!IsEmpty(param_norm)) {
        p.grad().mul_(clip_coef);
      }
    }
  }
}

}  // namespace utils
}  // namespace radish