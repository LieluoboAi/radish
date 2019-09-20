/*
 * File: radam_optimization.h
 * Project: optimization
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-13 10:16:47
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
}  // namespace serialize
}  // namespace torch

namespace knlp {
namespace optim {

struct TORCH_API RAdamOptions {
  explicit RAdamOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(int64_t, warmup_steps)=1;
};

class TORCH_API RAdam : public ::torch::optim::Optimizer {
 public:
  template <typename ParameterContainer>
  RAdam(ParameterContainer&& parameters, const RAdamOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {
    p_inf_ = 2.0 / (1.0 - options.beta2_) - 1.0;
  }

  void step() override;

  void save(::torch::serialize::OutputArchive& archive) const override;
  void load(::torch::serialize::InputArchive& archive) override;

  RAdamOptions options;

  std::vector<int64_t> step_buffers;
  std::vector<::torch::Tensor> exp_average_buffers;
  std::vector<::torch::Tensor> exp_average_sq_buffers;
  std::vector<::torch::Tensor> max_exp_average_sq_buffers;
  double p_inf_;

 private:
  RAdam() : options(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE(step_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_sq_buffers);
    _TORCH_OPTIM_SERIALIZE(max_exp_average_sq_buffers);
  }
};
}  // namespace optim
}  // namespace knlp
