/*
 * File: lamb.h
 * Project: optimization
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-29 10:13:42
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <utility>
#include <vector>

#include "torch/arg.h"
#include "torch/nn/module.h"
#include "torch/optim/optimizer.h"
#include "torch/optim/serialize.h"
#include "utils/logging.h"

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
}  // namespace serialize
}  // namespace torch

namespace radish {
namespace optim {

struct TORCH_API LambOptions {
  explicit LambOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, weight_decay) = 0.01;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, clip_norm) = 3.0;
};

class TORCH_API Lamb : public ::torch::optim::Optimizer {
 public:
  Lamb(std::vector<torch::Tensor> parameters, std::vector<std::string> names,
       const LambOptions& options);

  void step() override;

  void save(::torch::serialize::OutputArchive& archive) const override;
  void load(::torch::serialize::InputArchive& archive) override;

  LambOptions options;

  std::vector<int64_t> step_buffers;
  std::vector<::torch::Tensor> exp_average_buffers;
  std::vector<::torch::Tensor> exp_average_sq_buffers;

 private:
  Lamb() : options(0) {}
  std::vector<std::string> names_;
  std::vector<bool> need_weight_decay_;

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE(step_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_sq_buffers);
  }
};
}  // namespace optim
}  // namespace radish