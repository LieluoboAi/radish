/*
 * File: span_bert_example.h
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-20 10:38:24
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once

#include "train/data/llb_example.h"

namespace knlp {
using Tensor = torch::Tensor;
class SpanBertExample : public knlp::data::LlbExample {
 public:
  std::vector<Tensor> ToTensorList() override;
  bool FromMessage(const train::TrainExample& ex) override;

 private:
  Tensor span_left_;
  Tensor span_right_;
  Tensor target_indexies_;
  Tensor target_label_;
};

}  // namespace knlp
