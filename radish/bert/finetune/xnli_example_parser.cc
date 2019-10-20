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
#include "radish/utils/logging.h"
#include "radish/utils/text_tokenizer.h"

namespace radish {
static int kMaxLen = 128;

XNLIExampleParser::XNLIExampleParser() : gen_(std::random_device{}()) {}

XNLIExampleParser::~XNLIExampleParser() = default;
bool XNLIExampleParser::Init(const Json::Value& config) {
  std::string tokenizer_vocab = config.get("tokenizer_vocab", "").asString();
  spdlog::info("got tokenizer_vocab:{}", tokenizer_vocab);
  if (tokenizer_vocab.empty()) {
    return false;
  }
  std::string tokenizer_cls =
      config.get("tokenizer_class", "radish::BertTokenizer").asString();
  tokenizer_.reset(radish::TextTokenizerFactory::Create(tokenizer_cls));
  if (!tokenizer_.get()) {
    spdlog::info("Can't get tokenizer for cls:{}", tokenizer_cls);
    return false;
  }
  if (!tokenizer_->Init(tokenizer_vocab)) {
    return false;
  }
  return true;
}

void XNLIExampleParser::_select_a_b_ids(std::vector<int>& aids,
                                        std::vector<int>& bids,
                                        bool needRandom) {
  int alen = aids.size();
  int blen = bids.size();
  if (alen + blen <= kMaxLen - 3) {
    return;
  }
  int mustKeep = (kMaxLen - 3) / 3;
  if (alen <= mustKeep) {
    blen = kMaxLen - 3 - alen;
    bids.erase(bids.begin() + blen, bids.end());
  } else if (blen <= mustKeep) {
    alen = kMaxLen - 3 - blen;
    aids.erase(aids.begin() + alen, aids.end());
  } else {
    int remainKeep = kMaxLen - 3 - mustKeep * 2;
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
  auto aids = tokenizer_->Encode(ax);
  auto bids = tokenizer_->Encode(bx);
  Ex ex(kMaxLen);
  int clsId = tokenizer_->ClsId();
  int maskId = tokenizer_->MaskId();
  int sepId = tokenizer_->SepId();
  ex.x[0] = clsId;
  ex.types[0] = 0;
  std::string gold = nf == 3 ? ss[2] : ss[1];
  auto it = kGoldLabelMap.find(gold);
  CHECK(it != kGoldLabelMap.end()) << gold;
  ex.label = it->second;
  _select_a_b_ids(aids, bids, true);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 3);
  int k = 1;
  for (size_t i = 0; i < aids.size(); i++) {
    ex.x[k] = aids[i];
    ex.types[k] = 0;
    k += 1;
  }
  ex.types[k] = 0;
  ex.x[k] = sepId;
  k += 1;
  for (size_t i = 0; i < bids.size(); i++) {
    ex.x[k] = bids[i];
    ex.types[k] = 1;
    k += 1;
  }
  ex.x[k] = sepId;
  for (; k < kMaxLen; k++) {
    ex.types[k] = 1;
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
  auto aids = tokenizer_->Encode(a);
  auto bids = tokenizer_->Encode(b);
  Ex ex(kMaxLen);
  int clsId = tokenizer_->ClsId();
  int maskId = tokenizer_->MaskId();
  int sepId = tokenizer_->SepId();
  ex.x[0] = clsId;
  ex.types[0] = 0;
  _select_a_b_ids(aids, bids, false);
  CHECK_LE(aids.size() + bids.size(), kMaxLen - 3);
  int k = 1;
  for (size_t i = 0; i < aids.size(); i++) {
    ex.x[k] = aids[i];
    ex.types[k] = 0;
    k += 1;
  }
  ex.types[k] = 0;
  ex.x[k] = sepId;
  k += 1;
  for (size_t i = 0; i < bids.size(); i++) {
    ex.x[k] = bids[i];
    ex.types[k] = 1;
    k += 1;
  }
  ex.x[k] = sepId;
  for (; k < kMaxLen; k++) {
    ex.types[k] = 1;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.types, at::dtype(torch::kInt64).requires_grad(false)));
  return true;
}
}  // namespace radish
