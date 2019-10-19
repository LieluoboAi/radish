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

namespace radish {
using UString = std::basic_string<uint16_t>;
class BertTokenizer {
 public:
  explicit BertTokenizer(std::string vocab_file);
  virtual ~BertTokenizer();
  std::vector<int> Encode(std::string text);
  int Word2Id(std::string word) const;
  std::string Id2Word(int id) const;

 private:
  void max_seg_(std::string s, std::vector<int>& results);
  void load_vocab_(std::string path);
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
