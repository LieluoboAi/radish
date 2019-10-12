/*
 * File: query_same_parser.h
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-05 9:36:25
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
// 100 -1

struct QuerySameEx {
  explicit QuerySameEx(int maxLen) : x(maxLen, 0), types(maxLen, 0) {}
  std::vector<int> x;
  std::vector<int> types;
  // 是相似, 0 不相似， 1，相似
  int same;
};
using Tensor = torch::Tensor;
class QSExampleParser : public radish::data::ExampleParser {
 public:
  QSExampleParser();
  virtual ~QSExampleParser();
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
