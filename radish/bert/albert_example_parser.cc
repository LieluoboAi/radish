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

#include "radish/bert/albert_example_parser.h"

#include <algorithm>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "radish/utils/logging.h"
#include "radish/utils/text_tokenizer.h"

namespace radish {
static int kMaxLen = 200;
static int kMaxLabel = 28;

ALBertExampleParser::ALBertExampleParser()
    : gen_(std::random_device{}()),
      len_dist_({6, 3, 2}),
      random_p_dist_(0, 1) {}

bool ALBertExampleParser::_mask_seq(int maskId, int sepId, int clsId, int len,
                                    Ex& ex) {
  std::vector<int> offs;
  for (int i = 0; i < len; i += 4) {
    offs.push_back(i);
  }
  std::shuffle(offs.begin(), offs.end(), gen_);
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
      if ((i + off) >= len || masked[i + off] || ex.x[i + off] == sepId) {
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
  for (int i = 0; i < len; i++) {
    if (masked[i]) {
      int toReplace = 0;
      float p = random_p_dist_(gen_);
      float p2 = random_p_dist_(gen_);
      if (p > 0.8) {
        if (p2 <= 0.5) {
          toReplace = random_id_dist_(gen_);
          if (toReplace == sepId || toReplace == clsId) {
            toReplace = maskId;
          }
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
  random_id_dist_ =
      std::uniform_int_distribution<>(1, tokenizer_->TotalSize() - 1);
  return true;
}

void ALBertExampleParser::_select_a_b_ids(std::vector<int>& aids,
                                          std::vector<int>& bids) {
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
      std::uniform_int_distribution<> ran_keep_p(0, remainKeep);
      int keepa = ran_keep_p(gen_);
      int keepb = remainKeep - keepa;
      aids.erase(aids.begin() + mustKeep + keepa, aids.end());
      bids.erase(bids.begin() + mustKeep + keepb, bids.end());
    }
  }
}

bool ALBertExampleParser::ParseOne(std::string line,
                                   data::LlbExample& example) {
  std::string x = absl::AsciiStrToLower(line);
  std::vector<std::string> ss = absl::StrSplit(x, '\t');
  if (ss.size() != 3) {
    spdlog::warn("Opps , no \\t,{}", x);
    return false;
  }
  int target = atoi(ss[0].c_str());

  std::string ax = ss[1];
  std::string bx = ss[2];
  auto aids = tokenizer_->Encode(ax);
  auto bids = tokenizer_->Encode(bx);
  Ex ex(kMaxLen);
  int clsId = tokenizer_->ClsId();
  int maskId = tokenizer_->MaskId();
  int sepId = tokenizer_->SepId();
  ex.x[0] = clsId;
  ex.types[0] = 0;
  ex.ordered = target;
  _select_a_b_ids(aids, bids);
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
  // mid has place SEP
  for (size_t i = 0; i < bids.size(); i++) {
    ex.x[k] = bids[i];
    ex.types[k] = 1;
    k += 1;
  }
  ex.x[k] = sepId;
  for (; k < kMaxLen; k++) {
    ex.types[k] = 1;
  }
  if (!_mask_seq(maskId, sepId, clsId, k, ex)) {
    spdlog::warn("mask example error");
    return false;
  }
  example.features.push_back(
      torch::tensor(ex.x, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(torch::tensor(
      ex.indexies, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.types, at::dtype(torch::kInt64).requires_grad(false)));
  example.features.push_back(
      torch::tensor(ex.ordered, at::dtype(torch::kInt64).requires_grad(false)));
  example.target =
      torch::tensor(ex.target, at::dtype(torch::kInt64).requires_grad(false));
  return true;
}
}  // namespace radish
