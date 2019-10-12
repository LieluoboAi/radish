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

#include "radish/optimization/radam.h"

#include <cmath>
#include <functional>

#include "absl/strings/ascii.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/nn/module.h"
#include "torch/serialize/archive.h"
#include "torch/utils.h"

#include "ATen/ATen.h"
#include "radish/utils/tensor_util.h"

namespace radish {
namespace optim {

using Tensor = ::torch::Tensor;

RAdamOptions::RAdamOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

RAdam::RAdam(std::vector<torch::Tensor> parameters,
             std::vector<std::string> names, const RAdamOptions& options)
    : Optimizer(parameters), options(options), names_(names) {
  p_inf_ = 2.0 / (1.0 - options.beta2()) - 1.0;
  CHECK_EQ(names_.size(), parameters_.size());
  need_weight_decay_.resize(names_.size());
  for (size_t i = 0; i < names_.size(); i++) {
    std::string lname = absl::AsciiStrToLower(names_[i]);
    if (lname.find("bias") != std::string::npos ||
        lname.find("norm") != std::string::npos) {
      need_weight_decay_[i] = false;
      // spdlog::warn("do not wd for :{}", names_[i]);
    } else {
      need_weight_decay_[i] = true;
    }
  }
}

void RAdam::step() {
  // 先clip下梯度
  if (options.clip_norm() > options.eps()) {
    radish::utils::ClipGradienNorm(parameters_, options.clip_norm());
  }
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    bool need_weight_decay = need_weight_decay_.at(i);
    if (!p.grad().defined() || !p.requires_grad()) {
      continue;
    }
    auto lr = options.learning_rate();
    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);
    buffer_at(step_buffers, i) += 1;
    if (buffer_at(step_buffers, i) < options.warmup_steps()) {
      lr *= buffer_at(step_buffers, i) / (options.warmup_steps() + 0.0001);
    }
    float beta2_t = std::pow(options.beta2(), buffer_at(step_buffers, i));
    float beta1_t = std::pow(options.beta1(), buffer_at(step_buffers, i));

    exp_average.mul_(options.beta1()).add_(p.grad(), 1 - options.beta1());
    exp_average_sq.mul_(options.beta2())
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2());

    const auto pt =
        p_inf_ - (2.0 * buffer_at(step_buffers, i) * beta2_t) / (1 - beta2_t);
    if (options.weight_decay() > 0 && need_weight_decay) {
      torch::NoGradGuard guard;
      p.add_(p, -options.weight_decay() * lr);
    }

    if (pt > 5.0) {
      const auto denorm = exp_average_sq.sqrt().add_(options.eps());
      double r =
          ((pt - 4) * (pt - 2) * p_inf_) / ((p_inf_ - 4) * (p_inf_ - 2) * pt);
      r = sqrt(r);
      torch::NoGradGuard guard;
      const auto step_size = (lr * r) / (1.0 - beta1_t);
      p.add_(torch::div(exp_average, denorm), -step_size);
    } else {
      torch::NoGradGuard guard;
      const auto step_size = lr / (1.0 - beta1_t);
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
