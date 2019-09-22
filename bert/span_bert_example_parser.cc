/*
 * File: span_bert_example.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-20 10:52:21
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "bert/span_bert_example_parser.h"

#include "absl/strings/ascii.h"
#include "sentencepiece/sentencepiece_processor.h"

#include "glog/logging.h"

namespace knlp {
// 200 -2
static int kMaxLen = 198;
static int kMaxLabel = 28;

struct Ex {
  Ex()
      : x(kMaxLen + 2, 0),
        target(kMaxLabel, 0),
        indexies(kMaxLabel, 0),
        spanLeft(kMaxLabel, 0),
        spanRight(kMaxLabel, 0) {}
  std::vector<int> x;
  std::vector<int> target;
  std::vector<int> indexies;
  std::vector<int> spanLeft;
  std::vector<int> spanRight;
};
SpanBertExampleParser::~SpanBertExampleParser() = default;
bool SpanBertExampleParser::Init(const Json::Value& config) {
  std::string spm_model_path = config.get("spm_model_path", "").asString();
  if (spm_model_path.empty()) {
    return false;
  }
  spp_.reset(new sentencepiece::SentencePieceProcessor());
  if (!spp_->Load(spm_model_path).ok()) {
    return false;
  }
  return true;
}

bool SpanBertExampleParser::ParseOne(train::TrainExample& protoData,
                                     data::LlbExample& example) {
  auto stringMap = protoData.string_feature();
  auto it = stringMap.find("x");
  if (it == stringMap.end()) {
    VLOG(0) << "no feature 'x'";
    return false;
  }
  std::string x = absl::AsciiStrToLower(it->second);
  auto ids = spp_->EncodeAsIds(x);

  return true;
}
}  // namespace knlp
