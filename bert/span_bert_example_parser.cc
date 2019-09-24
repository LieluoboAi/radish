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
#include "utils/logging.h"

namespace radish {
// 200 -2
static int kMaxLen = 198;
static int kMaxLabel = 28;

SpanBertExampleParser::SpanBertExampleParser() {
  std::random_device device;
  gen_.reset(new std::mt19937(device()));
}

bool SpanBertExampleParser::_mask_seq(int maskId, int totalVocabSize, int len,
                                      Ex& ex) {
  std::vector<int> offs;
  for (int i = 0; i < len; i += 4) {
    offs.push_back(i);
  }
  std::discrete_distribution<> dist({10, 15, 20, 20, 20, 15});
  std::uniform_int_distribution<> randomId(1, totalVocabSize - 1);
  std::uniform_real_distribution<> randomP(0, 1.0);
  std::random_shuffle(offs.begin(), offs.end());
  std::vector<bool> masked(len, false);
  int num_masked = 0;
  for (auto p : offs) {
    int off = p * 4;
    if (num_masked >= kMaxLabel) {
      break;
    }
    int drawLen = dist(*gen_) + 1;
    if (drawLen + num_masked > kMaxLabel) {
      continue;
    }
    if (drawLen < 4) {
      std::uniform_int_distribution<> uid(0, 5 - drawLen);
      off += uid(*gen_);
    }
    if (off == 0) {
      continue;
    }
    bool valid = true;
    for (int i = 0; i < drawLen; i++) {
      if (masked[i + off] || (i + off) >= (len - 1)) {
        valid = false;
        break;
      }
    }
    // actually mask
    if (valid) {
      for (int i = 0; i < drawLen; i++) {
        masked[i + off] = true;
      }
      num_masked += drawLen;
    }
  }
  int start = -1;
  for (int i = 0; i < len; i++) {
    if (masked[i]) {
      if (start == -1) {
        start = i;
      }
    } else {
      if (start != -1) {
        CHECK_GT(start, 0);
        int toReplace = 0;
        float p = randomP(*gen_);
        float p2 = randomP(*gen_);
        if (p > 0.8) {
          if (p2 <= 0.5) {
            toReplace = randomId(*gen_);
          }
        } else {
          toReplace = maskId;
        }
        for (int k = start; k < i; k++) {
          ex.target.push_back(ex.x[k]);
          ex.indexies.push_back(k);
          ex.spanLeft.push_back(start - 1);
          ex.spanRight.push_back(i);
          if (toReplace != 0) {
            ex.x[k] = toReplace;
          }
        }
      }
      start = -1;
    }
  }
  CHECK_LE(static_cast<int>(ex.target.size()), kMaxLabel);
  for (int i = ex.target.size(); i < kMaxLabel; i++) {
    ex.target.push_back(0);
    ex.indexies.push_back(0);
    ex.spanLeft.push_back(0);
    ex.spanRight.push_back(0);
  }
  return true;
}
SpanBertExampleParser::~SpanBertExampleParser() = default;
bool SpanBertExampleParser::Init(const Json::Value& config) {
  std::string spm_model_path = config.get("spm_model_path", "").asString();
  spdlog::info("got spm_model_path:{}", spm_model_path);
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
    spdlog::warn("no feature 'x'");
    return false;
  }
  std::string x = absl::AsciiStrToLower(it->second);
  auto ids = spp_->EncodeAsIds(x);
  int totalVocabSize = spp_->GetPieceSize();
  Ex ex(kMaxLen + 2);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  if (ids.size() < 2) {
    spdlog::warn("too short to be an example:{}", ids.size());
    return false;
  }
  ex.x[0] = clsId;
  int i = 1;
  for (; i <= static_cast<int>(ids.size()) && i <= kMaxLen; i++) {
    ex.x[i] = ids[i - 1];
  }
  ex.x[i] = sepId;
  if (!_mask_seq(maskId, totalVocabSize, i, ex)) {
    spdlog::warn("mask example error");
    return false;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false))
          .clone());
  example.features.push_back(
      torch::tensor(ex.indexies, at::dtype(torch::kInt64).requires_grad(false))
          .clone());
  example.features.push_back(
      torch::tensor(ex.spanLeft, at::dtype(torch::kInt64).requires_grad(false))
          .clone());
  example.features.push_back(
      torch::tensor(ex.spanRight, at::dtype(torch::kInt64).requires_grad(false))
          .clone());
  example.target =
      torch::tensor(ex.target, at::dtype(torch::kInt64).requires_grad(false))
          .clone();
  return true;
}
}  // namespace radish
