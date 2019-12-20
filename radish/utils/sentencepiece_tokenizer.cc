/*
 * File: sentencepiece_tokenizer.cc
 * Project: utils
 * File Created: Sunday, 20th October 2019 1:10:47 pm
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 1:10:49 pm
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */
#include "radish/utils/sentencepiece_tokenizer.h"

#include <algorithm>

#include "sentencepiece/sentencepiece_processor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/strip.h"

namespace radish {

std::string SentencePieceTokenizer::kUnkToken = "[UNK]";
std::string SentencePieceTokenizer::kMaskToken = "[MASK]";
std::string SentencePieceTokenizer::kSepToken = "[SEP]";
std::string SentencePieceTokenizer::kPadToken = "[PAD]";
std::string SentencePieceTokenizer::kClsToken = "[CLS]";

bool SentencePieceTokenizer::Init(std::string vocab_file) {
  spp_.reset(new sentencepiece::SentencePieceProcessor());
  if (!spp_->Load(vocab_file).ok()) {
    return false;
  }
  return true;
}
std::vector<int> SentencePieceTokenizer::Encode(std::string text) {
  (void)s_bRegistered;  // force the registeration
  std::vector<int> results;
  absl::RemoveExtraAsciiWhitespace(&text);
  text = absl::AsciiStrToLower(text);
  std::vector<int> ids = spp_->EncodeAsIds(text);
  std::replace_if(ids.begin(), ids.end(), [](int v) { return v == 0; },
                  UnkId());
  return results;
}

int SentencePieceTokenizer::PadId() const { return 0; }
int SentencePieceTokenizer::MaskId() const { return spp_->GetPieceSize() + 1; }
int SentencePieceTokenizer::SepId() const { return spp_->GetPieceSize() + 2; }
int SentencePieceTokenizer::ClsId() const { return spp_->GetPieceSize(); }
int SentencePieceTokenizer::UnkId() const { return spp_->GetPieceSize() + 3; }

int SentencePieceTokenizer::TotalSize() const{
  return spp_->GetPieceSize() + 4;
}
int SentencePieceTokenizer::Word2Id(std::string s) const {
  if (s == kPadToken) {
    return 0;
  } else if (s == kMaskToken) {
    return MaskId();
  } else if (s == kSepToken) {
    return SepId();
  } else if (s == kClsToken) {
    return ClsId();
  } else if (s == kUnkToken) {
    return UnkId();
  }
  int id = spp_->PieceToId(s);
  if (id == 0) {
    return UnkId();
  }
  return id;
}
std::string SentencePieceTokenizer::Id2Word(int id) const {
  if (id == 0) {
    return kPadToken;
  } else if (id == MaskId()) {
    return kMaskToken;
  } else if (id == SepId()) {
    return kSepToken;
  } else if (id == ClsId()) {
    return kClsToken;
  } else if (id == UnkId()) {
    return kUnkToken;
  }
  return spp_->IdToPiece(id);
}

}  // namespace radish
