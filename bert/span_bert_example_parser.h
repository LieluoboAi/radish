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
#include "train/data/example_parser.h"

namespace sentencepiece {
class SentencePieceProcessor;
}  // namespace sentencepiece

namespace radish {
using Tensor = torch::Tensor;
class SpanBertExampleParser : public radish::data::ExampleParser {
 public:
  virtual ~SpanBertExampleParser();
  bool Init(const Json::Value& config) override;
  bool ParseOne(train::TrainExample& protoData,
                data::LlbExample& example) override;

 private:
  std::shared_ptr<sentencepiece::SentencePieceProcessor> spp_;
};

}  // namespace radish
