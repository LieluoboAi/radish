/*
 * File: albert_example_parser.h
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-27 4:39:37
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
// 200 -1

struct Ex {
  explicit Ex(int maxLen) : x(maxLen, 0), types(maxLen,0) {}
  std::vector<int> x;
  std::vector<int> types;
  std::vector<int> target;
  std::vector<int> indexies;
  // 是否有序的, 0 乱序， 1，有序
  int ordered;
};
using Tensor = torch::Tensor;
class ALBertExampleParser : public radish::data::ExampleParser {
 public:
  ALBertExampleParser();
  virtual ~ALBertExampleParser();
  bool Init(const Json::Value& config) override;
  bool ParseOne(std::string line,
                data::LlbExample& example) override;

 private:
  bool _mask_seq(int maskId, int seqId, int totalVocabSize, int len, Ex& ex);
  void _select_a_b_ids(std::vector<int>& aids, std::vector<int>& bids);
  std::shared_ptr<sentencepiece::SentencePieceProcessor> spp_;
  std::mt19937 gen_;
  std::discrete_distribution<> len_dist_;
  std::uniform_int_distribution<> random_id_dist_;
  std::uniform_real_distribution<> random_p_dist_;
};

}  // namespace radish

