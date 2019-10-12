/*
 * File: xnli_example_parser.h
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-11 9:25:39
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <memory>
#include <random>
#include "radish/train/data/example_parser.h"

namespace sentencepiece {
class SentencePieceProcessor;
}  // namespace sentencepiece

namespace radish {
// 128 -1

struct Ex {
  explicit Ex(int maxLen) : x(maxLen, 0), types(maxLen, 0) {}
  std::vector<int> x;
  std::vector<int> types;
  //  0 中立， 1，蕴涵， 2，矛盾
  int label;
};
using Tensor = torch::Tensor;
class XNLIExampleParser : public radish::data::ExampleParser {
 public:
  XNLIExampleParser();
  virtual ~XNLIExampleParser();
  bool Init(const Json::Value& config) override;
  bool ParseOne(std::string line, data::LlbExample& example) override;
  // For inference
  bool CreateNoLabel(const std::string& a, const std::string& b,
                     data::LlbExample& example);

 private:
  void _select_a_b_ids(std::vector<int>& aids, std::vector<int>& bids,
                       bool needRandom = false);
  std::shared_ptr<sentencepiece::SentencePieceProcessor> spp_;
  std::mt19937 gen_;
};

}  // namespace radish
