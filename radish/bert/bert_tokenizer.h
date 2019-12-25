/*
 * File: bert_tokenizer.h
 * Project: bert
 * File Created: Saturday, 19th October 2019 10:41:25 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Saturday, 19th October 2019 10:41:38 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "radish/utils/text_tokenizer.h"


namespace radish {
using UString = std::basic_string<uint16_t>;
class BertTokenizer : public TextTokenizer,
                      TextTokenizerRegisteeStub<BertTokenizer> {
 public:
  bool Init(std::string vocab) override;
  bool InitByFileContent(std::string content);
  std::vector<int> Encode(std::string text) override;
  int Word2Id(std::string word) const override;
  std::string Id2Word(int id) const override;
  int PadId() const override;
  int MaskId() const override;
  int SepId() const override;
  int ClsId() const override;
  int UnkId() const override;
  int TotalSize() const override;
 private:
  void max_seg_(std::string s, std::vector<int>& results);
  void load_vocab_(std::string path, std::vector<std::string>& lines);
  void init_from_lines(const std::vector<std::string>& lines);
  UString _basic_tokenize(UString text);
  UString _clean(UString text);
  std::unordered_map<std::string, int> token_2_id_map_;
  std::vector<std::string> tokens_;
  static std::string kUnkToken;
  static std::string kMaskToken;
  static std::string kSepToken;
  static std::string kPadToken;
  static std::string kClsToken;
};
}  // namespace radish
