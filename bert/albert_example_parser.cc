/*
 * File: albert_example_parser.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-27 4:56:56
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "bert/albert_example_parser.h"

#include "absl/strings/ascii.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "utils/logging.h"

namespace radish {
// 200 -1
static int kMaxLen = 199;
static int kMaxLabel = 28;

ALBertExampleParser::ALBertExampleParser()
    : gen_(std::random_device{}()),
      len_dist_({12, 6, 4, 3}),
      random_p_dist_(0, 1) {}

bool ALBertExampleParser::_mask_seq(int maskId, int sepId, int totalVocabSize,
                                    int len, Ex& ex) {
  std::vector<int> offs;
  for (int i = 0; i < len; i += 4) {
    offs.push_back(i);
  }
  std::random_shuffle(offs.begin(), offs.end());
  std::vector<bool> masked(len, false);
  int num_masked = 0;
  for (auto p : offs) {
    int off = p * 4;
    if (num_masked >= kMaxLabel) {
      break;
    }
    int drawLen = len_dist_(gen_) + 1;
    if (drawLen + num_masked > kMaxLabel) {
      continue;
    }
    std::uniform_int_distribution<> off_dist(0, 5 - drawLen);
    off += off_dist(gen_);

    if (off == 0) {
      // 第一个位置留给 CLS, 不mask
      continue;
    }
    bool valid = true;
    for (int i = 0; i < drawLen; i++) {
      if (masked[i + off] || ex.x[i + off] == sepId || (i + off) >= len) {
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
      int toReplace = 0;
      float p = random_p_dist_(gen_);
      float p2 = random_p_dist_(gen_);
      if (p > 0.8) {
        if (p2 <= 0.5) {
          toReplace = random_id_dist_(gen_);
        }
      } else {
        toReplace = maskId;
      }
      ex.target.push_back(ex.x[i]);
      ex.indexies.push_back(i);
      if (toReplace != 0) {
        ex.x[i] = toReplace;
      }
    }
  }
  CHECK_LE(static_cast<int>(ex.target.size()), kMaxLabel);
  for (int i = ex.target.size(); i < kMaxLabel; i++) {
    ex.target.push_back(0);
    ex.indexies.push_back(0);
  }
  return true;
}
ALBertExampleParser::~ALBertExampleParser() = default;
bool ALBertExampleParser::Init(const Json::Value& config) {
  std::string spm_model_path = config.get("spm_model_path", "").asString();
  spdlog::info("got spm_model_path:{}", spm_model_path);
  if (spm_model_path.empty()) {
    return false;
  }
  spp_.reset(new sentencepiece::SentencePieceProcessor());
  if (!spp_->Load(spm_model_path).ok()) {
    return false;
  }
  int totalVocab = spp_->GetPieceSize();
  random_id_dist_ = std::uniform_int_distribution<>(1, totalVocab - 1);
  return true;
}

bool ALBertExampleParser::ParseOne(std::string line,
                                   data::LlbExample& example) {
  absl::RemoveExtraAsciiWhitespace(&line);
  std::string x = absl::AsciiStrToLower(line);
  auto ids = spp_->EncodeAsIds(x);
  int totalVocabSize = spp_->GetPieceSize();
  Ex ex(kMaxLen + 1);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  if (ids.size() < 10) {
    spdlog::warn("too short to be an example:{}", ids.size());
    return false;
  }
  ex.x[0] = clsId;
  if (random_p_dist_(gen_) <= 0.5) {
    ex.ordered = true;
  } else {
    ex.ordered = false;
  }
  int total = ids.size();
  int off = 0;
  int mid = total / 2;
  int end = total;
  if (total > kMaxLen - 2) {
    std::discrete_distribution<> ranoff_p(0, total - kMaxLen);
    off += ranoff_p(gen_);
    mid = off + kMaxLen / 2 - 1;
    end = off + kMaxLen - 2;
  }
  int k = 1;
  if (ex.ordered) {
    for (int i = off; i < mid; i++) {
      ex.x[k] = ids[i];
      k += 1;
    }
    ex.a_len_ = k;
    ex.x[k] = sepId;
    k += 1;
    for (int i = mid; i < end; i++) {
      ex.x[k] = ids[i];
      k += 1;
    }
    ex.t_len_ = k;
  } else {
    for (int i = mid; i < end; i++) {
      ex.x[k] = ids[i];
      k += 1;
    }
    ex.a_len_ = k;
    ex.x[k] = sepId;
    k += 1;
    for (int i = off; i < mid; i++) {
      ex.x[k] = ids[i];
      k += 1;
    }
    ex.t_len_ = k;
  }

  if (!_mask_seq(maskId, sepId, totalVocabSize, k, ex)) {
    spdlog::warn("mask example error");
    return false;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(torch::tensor(
      ex.indexies, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.a_len_, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.t_len_, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.ordered, at::dtype(torch::kInt32).requires_grad(false)));
  example.target =
      torch::tensor(ex.target, at::dtype(torch::kInt64).requires_grad(false));
  return true;
}
}  // namespace radish
