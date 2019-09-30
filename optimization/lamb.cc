/*
 * File: lamb.cc
 * Project: optimization
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-29 10:19:10
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "optimization/lamb.h"

#include <cmath>
#include <functional>

#include "absl/strings/ascii.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/nn/module.h"
#include "torch/serialize/archive.h"
#include "torch/utils.h"

#include "ATen/ATen.h"
#include "utils/tensor_util.h"

namespace radish {
namespace optim {

using Tensor = ::torch::Tensor;

LambOptions::LambOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

Lamb::Lamb(std::vector<torch::Tensor> parameters,
           std::vector<std::string> names, const LambOptions& options)
    : Optimizer(parameters), options(options), names_(names) {
  CHECK_EQ(names_.size(), parameters_.size());
  need_weight_decay_.resize(names_.size());
  for (size_t i = 0; i < names_.size(); i++) {
    std::string lname = absl::AsciiStrToLower(names_[i]);
    if (lname.find("bias") != std::string::npos ||
        lname.find("norm") != std::string::npos) {
      need_weight_decay_[i] = false;
    } else {
      need_weight_decay_[i] = true;
    }
  }
}

void Lamb::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    bool need_weight_decay = need_weight_decay_.at(i);
    if (!p.grad().defined()) {
      continue;
    }
    auto lr = options.learning_rate_;
    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);
    buffer_at(step_buffers, i) += 1;
    float beta2_t = std::pow(options.beta2_, buffer_at(step_buffers, i));
    float beta1_t = std::pow(options.beta1_, buffer_at(step_buffers, i));

    exp_average.mul_(options.beta1_).add_(p.grad(), 1 - options.beta1_);
    exp_average_sq.mul_(options.beta2_)
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2_);

    auto adam_step = exp_average.div(exp_average_sq.sqrt().add(options.eps_));
    if (options.weight_decay_ > 0 && need_weight_decay) {
      adam_step.add_(p, options.weight_decay_);
    }
    {
      torch::NoGradGuard guard;
      auto adam_norm = adam_step.pow(2).sum().sqrt().item().to<float>();
      auto weight_norm =
          p.pow(2).sum().sqrt_().clamp_(0, 10).item().to<float>();
      float trust_ratio = weight_norm / adam_norm;
      if ((weight_norm < 1e-8 && weight_norm > -1e-8) ||
          (adam_norm < 1e-8 && adam_norm > -1e-8)) {
        trust_ratio = 1.0;
      }
      p.add_(adam_step, -lr * trust_ratio);
    }
  }
}

void Lamb::save(::torch::serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Lamb::load(::torch::serialize::InputArchive& archive) {
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
