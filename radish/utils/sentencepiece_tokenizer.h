/*
 * File: sentencepiece_tokenizer.h
 * Project: utils
 * File Created: Sunday, 20th October 2019 1:05:16 pm
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 1:05:18 pm
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "radish/utils/text_tokenizer.h"

namespace sentencepiece {
class SentencePieceProcessor;
}  // namespace sentencepiece

namespace radish {
class SentencePieceTokenizer
    : public TextTokenizer,
      TextTokenizerRegisteeStub<SentencePieceTokenizer> {
 public:
  bool Init(std::string vocab) override;
  std::vector<int> Encode(std::string text) override;
  int Word2Id(std::string word) const override;
  std::string Id2Word(int id) const override;
  int PadId() const override;
  int MaskId() const override;
  int SepId() const override;
  int ClsId() const override;
  int UnkId() const override;
  int TotalSize() const override;
  static std::string kUnkToken;
  static std::string kMaskToken;
  static std::string kSepToken;
  static std::string kPadToken;
  static std::string kClsToken;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spp_;
};
}  // namespace radish
