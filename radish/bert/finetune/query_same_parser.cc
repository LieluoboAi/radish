/*
 * File: query_same_parser.cc
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-05 9:51:03
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "radish/bert/finetune/query_same_parser.h"

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "radish/utils/logging.h"

namespace radish {
// 128 -1
static int kMaxLen = 127;

QSExampleParser::QSExampleParser() : gen_(std::random_device{}()) {}

QSExampleParser::~QSExampleParser() = default;
bool QSExampleParser::Init(const Json::Value& config) {
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

void QSExampleParser::_select_a_b_ids(std::vector<int>& aids,
                                      std::vector<int>& bids, bool needRandom) {
  int alen = aids.size();
  int blen = bids.size();
  if (alen + blen <= kMaxLen - 2) {
    return;
  }
  int mustKeep = (kMaxLen - 2) / 3;
  if (alen <= mustKeep) {
    blen = kMaxLen - 2 - alen;
    bids.erase(bids.begin() + blen, bids.end());
  } else if (blen <= mustKeep) {
    alen = kMaxLen - 2 - blen;
    aids.erase(aids.begin() + alen, aids.end());
  } else {
    int remainKeep = kMaxLen - 2 - mustKeep * 2;
    if (alen - mustKeep <= remainKeep) {
      // keep all a
      int keepb = remainKeep - alen + mustKeep;
      bids.erase(bids.begin() + mustKeep + keepb, bids.end());
    } else if (blen - mustKeep <= remainKeep) {
      int keepa = remainKeep - blen + mustKeep;
      aids.erase(aids.begin() + mustKeep + keepa, aids.end());
    } else {
      int keepa = remainKeep / 2;
      if (needRandom) {
        std::uniform_int_distribution<> ran_keep_p(0, remainKeep);
        keepa = ran_keep_p(gen_);
      }
      int keepb = remainKeep - keepa;
      aids.erase(aids.begin() + mustKeep + keepa, aids.end());
      bids.erase(bids.begin() + mustKeep + keepb, bids.end());
    }
  }
}

bool QSExampleParser::ParseOne(std::string line, data::LlbExample& example) {
  std::string x = absl::AsciiStrToLower(line);
  std::vector<std::string> ss = absl::StrSplit(x, '\t');
  if (ss.size() != 3) {
    spdlog::warn("Opps , example error:{}", x);
    return false;
  }
  std::string ax = ss[0];
  std::string bx = ss[1];

  auto aids = spp_->EncodeAsIds(ax);
  auto bids = spp_->EncodeAsIds(bx);
  int totalVocabSize = spp_->GetPieceSize();
  QuerySameEx ex(kMaxLen + 1);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  ex.x[0] = clsId;
  ex.types[0] = 1;
  ex.same = atoi(ss[2].c_str());
  _select_a_b_ids(aids, bids, true);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 2);
  int k = 1;
  for (size_t i = 0; i < aids.size(); i++) {
    ex.x[k] = aids[i];
    ex.types[k] = 1;
    k += 1;
  }
  ex.types[k] = 1;
  ex.x[k] = sepId;
  k += 1;
  for (size_t i = 0; i < bids.size(); i++) {
    ex.x[k] = bids[i];
    ex.types[k] = 2;
    k += 1;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.types, at::dtype(torch::kInt64).requires_grad(false)));
  example.target =
      torch::tensor(ex.same, at::dtype(torch::kInt64).requires_grad(false));
  return true;
}

bool QSExampleParser::CreateNoLabel(const std::string& a, const std::string& b,
                                    data::LlbExample& example) {
  auto aids = spp_->EncodeAsIds(absl::AsciiStrToLower(a));
  auto bids = spp_->EncodeAsIds(absl::AsciiStrToLower(b));
  int totalVocabSize = spp_->GetPieceSize();
  QuerySameEx ex(kMaxLen + 1);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  ex.x[0] = clsId;
  ex.types[0] = 1;
  _select_a_b_ids(aids, bids, false);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 2);
  int k = 1;
  for (size_t i = 0; i < aids.size(); i++) {
    ex.x[k] = aids[i];
    ex.types[k] = 1;
    k += 1;
  }
  ex.types[k] = 1;
  ex.x[k] = sepId;
  k += 1;
  for (size_t i = 0; i < bids.size(); i++) {
    ex.x[k] = bids[i];
    ex.types[k] = 2;
    k += 1;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.types, at::dtype(torch::kInt64).requires_grad(false)));
  return true;
}
}  // namespace radish
