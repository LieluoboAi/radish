/*
 * File: xnli_example_parser.cc
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-11 9:35:47
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "radish/bert/finetune/xnli_example_parser.h"

#include <unordered_map>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "radish/utils/logging.h"

namespace radish {
// 128 -1
static int kMaxLen = 127;

XNLIExampleParser::XNLIExampleParser() : gen_(std::random_device{}()) {}

XNLIExampleParser::~XNLIExampleParser() = default;
bool XNLIExampleParser::Init(const Json::Value& config) {
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

void XNLIExampleParser::_select_a_b_ids(std::vector<int>& aids,
                                        std::vector<int>& bids,
                                        bool needRandom) {
  int alen = aids.size();
  int blen = bids.size();
  if (alen + blen <= kMaxLen - 1) {
    return;
  }
  int mustKeep = (kMaxLen - 1) / 3;
  if (alen <= mustKeep) {
    blen = kMaxLen - 1 - alen;
    bids.erase(bids.begin() + blen, bids.end());
  } else if (blen <= mustKeep) {
    alen = kMaxLen - 1 - blen;
    aids.erase(aids.begin() + alen, aids.end());
  } else {
    int remainKeep = kMaxLen - 1 - mustKeep * 2;
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

static const std::unordered_map<std::string, int> kGoldLabelMap = {
    {"neutral", 0},
    {"entailment", 1},
    {"contradiction", 2},
    {"contradictory", 2}};

bool XNLIExampleParser::ParseOne(std::string line, data::LlbExample& example) {
  std::string x = absl::AsciiStrToLower(line);
  std::vector<std::string> ss = absl::StrSplit(x, absl::ByChar('\t'));
  int nf = ss.size();
  if (nf != 19 && nf != 3) {
    spdlog::warn("Opps , size is:{}, example error:{}", ss.size(), x);
    return false;
  }

  std::string ax = nf == 3 ? ss[0] : ss[6];
  std::string bx = nf == 3 ? ss[1] : ss[7];
  if (nf == 3) {
    ax = absl::StrReplaceAll(ax, {{" ", ""}});
    bx = absl::StrReplaceAll(bx, {{" ", ""}});
  }
  auto aids = spp_->EncodeAsIds(ax);
  auto bids = spp_->EncodeAsIds(bx);
  int totalVocabSize = spp_->GetPieceSize();
  Ex ex(kMaxLen + 1);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  ex.x[0] = clsId;
  ex.types[0] = 1;
  std::string gold = nf == 3 ? ss[2] : ss[1];
  auto it = kGoldLabelMap.find(gold);
  CHECK(it != kGoldLabelMap.end()) << gold;
  ex.label = it->second;
  _select_a_b_ids(aids, bids, true);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 1);
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
      torch::tensor(ex.label, at::dtype(torch::kInt64).requires_grad(false));
  return true;
}

bool XNLIExampleParser::CreateNoLabel(const std::string& a,
                                      const std::string& b,
                                      data::LlbExample& example) {
  auto aids = spp_->EncodeAsIds(absl::AsciiStrToLower(a));
  auto bids = spp_->EncodeAsIds(absl::AsciiStrToLower(b));
  int totalVocabSize = spp_->GetPieceSize();
  Ex ex(kMaxLen + 1);
  int clsId = totalVocabSize;
  int maskId = totalVocabSize + 1;
  int sepId = totalVocabSize + 2;
  ex.x[0] = clsId;
  ex.types[0] = 1;
  _select_a_b_ids(aids, bids, false);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 1);
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
