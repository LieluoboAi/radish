/*
 * File: tensor_util.h
 * Project: utils
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-25 10:11:33
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once
#include "torch/torch.h"

namespace radish {
namespace utils {
bool IsEmpty(const torch::Tensor& x);

void ClipGradienNorm(std::vector<at::Tensor>& parameters, float max_norm);

}  // namespace utils
}  // namespace radish
