/*
 * File: radam.cc
 * Project: optimization
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-14 8:59:06
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "optimization/radam.h"

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace radish {
namespace optim {

using Tensor = ::torch::Tensor;

RAdamOptions::RAdamOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

void RAdam::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }
    auto lr = options.learning_rate_;
    // maybe place at the end
    if (options.weight_decay_ > 0) {
      torch::NoGradGuard guard;
      p.grad() = p.grad() + options.weight_decay_ * p;
    }
    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);

    buffer_at(step_buffers, i) += 1;
    if (buffer_at(step_buffers, i) < options.warmup_steps_) {
      lr *= buffer_at(step_buffers, i) / (options.warmup_steps_ + 0.0001);
    }
    const auto bias_correction1 =
        1 - std::pow(options.beta1_, buffer_at(step_buffers, i));
    const auto bias_correction2 =
        1 - std::pow(options.beta2_, buffer_at(step_buffers, i));
    exp_average.mul_(options.beta1_).add_(p.grad(), 1 - options.beta1_);
    exp_average_sq.mul_(options.beta2_)
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2_);

    const auto pt = p_inf_ - (2.0 * buffer_at(step_buffers, i) *
                              (1.0 - bias_correction2) / bias_correction2);
    if (pt > 5.0) {
      const auto sq_correct = (exp_average_sq / bias_correction2).sqrt();
      double r =
          ((pt - 4) * (pt - 2) * p_inf_) / ((p_inf_ - 4) * (p_inf_ - 2) * pt);
      r = sqrt(r);
      torch::NoGradGuard guard;
      const auto step_size = (lr * r) / bias_correction1;
      p.addcdiv_(exp_average, sq_correct + options.eps_, -step_size);
    } else {
      torch::NoGradGuard guard;
      const auto step_size = lr / bias_correction1;
      p.add_(exp_average, -step_size);
    }
  }
}

void RAdam::save(::torch::serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void RAdam::load(::torch::serialize::InputArchive& archive) {
  serialize(*this, archive);
}

}  // namespace optim
}  // namespace radish

namespace torch {
namespace optim {
void serialize(serialize::OutputArchive& archive, const std::string& key,
               const std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(steps.size());
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }
  serialize(archive, key, tensors);
}

void serialize(serialize::InputArchive& archive, const std::string& key,
               std::vector<int64_t>& steps) {
  steps.clear();
  std::vector<torch::Tensor> tensors;
  serialize(archive, key, tensors);
  for (const auto& step : tensors) {
    steps.push_back(step.item<int64_t>());
  }
}
}  // namespace optim
}  // namespace torch
