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
#include <memory>
#include <random>
#include "train/data/example_parser.h"

namespace sentencepiece {
class SentencePieceProcessor;
}  // namespace sentencepiece

namespace radish {
// 200 -2

struct Ex {
  explicit Ex(int maxLen) : x(maxLen, 0) {}
  std::vector<int> x;
  std::vector<int> target;
  std::vector<int> indexies;
  std::vector<int> spanLeft;
  std::vector<int> spanRight;
};
using Tensor = torch::Tensor;
class SpanBertExampleParser : public radish::data::ExampleParser {
 public:
  SpanBertExampleParser();
  virtual ~SpanBertExampleParser();
  bool Init(const Json::Value& config) override;
  bool ParseOne(train::TrainExample& protoData,
                data::LlbExample& example) override;

 private:
  bool _mask_seq(int maskId, int totalVocabSize, int len, Ex& ex);
  std::shared_ptr<sentencepiece::SentencePieceProcessor> spp_;
  std::mt19937 gen_;
};

}  // namespace radish
