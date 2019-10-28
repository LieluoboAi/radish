/*
 * File: bert_tokenizer.cc
 * Project: bert
 * File Created: Saturday, 19th October 2019 11:26:14 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Saturday, 19th October 2019 11:26:17 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include "radish/bert/bert_tokenizer.h"

#include <cwctype>
#include <fstream>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "radish/utils/logging.h"
#include "utf8proc.h"
#include "source/utf8.h"

namespace radish {

std::string BertTokenizer::kUnkToken = "[UNK]";
std::string BertTokenizer::kMaskToken = "[MASK]";
std::string BertTokenizer::kSepToken = "[SEP]";
std::string BertTokenizer::kPadToken = "[PAD]";
std::string BertTokenizer::kClsToken = "[CLS]";
static std::unordered_set<uint16_t> kChinesePunts = {
    12290, 65306, 65311, 8212, 8216, 12304, 12305, 12298, 12299, 65307};
static int kMaxCharsPerWords = 100;

bool BertTokenizer::Init(std::string vocab_file) {
  load_vocab_(vocab_file);
  if (token_2_id_map_.find(kPadToken) == token_2_id_map_.end()) {
    return false;
  }
  if (token_2_id_map_.find(kUnkToken) == token_2_id_map_.end()) {
    return false;
  }
  if (token_2_id_map_.find(kClsToken) == token_2_id_map_.end()) {
    return false;
  }
  if (token_2_id_map_.find(kSepToken) == token_2_id_map_.end()) {
    return false;
  }
  if (token_2_id_map_.find(kMaskToken) == token_2_id_map_.end()) {
    return false;
  }
  int v = token_2_id_map_.at(kPadToken);
  if (v != 0) {
    return false;
  }
  return true;
}

std::vector<int> BertTokenizer::Encode(std::string text) {
  (void)s_bRegistered;  // force the registeration
  std::vector<int> results;
  absl::RemoveExtraAsciiWhitespace(&text);
  char* nfkcstr = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(text.c_str())));
  if (nfkcstr == nullptr) {
    spdlog::info("do NFD error");
    return {};
  }
  text.assign(nfkcstr, strlen(nfkcstr));
  free(nfkcstr);
  text = absl::AsciiStrToLower(text);
  UString unicodes;
  utf8::utf8to16(text.c_str(), text.c_str() + text.size(),
                 std::back_inserter(unicodes));
  unicodes = _strip_accents(unicodes);
  unicodes = _clean(unicodes);
  unicodes = _basic_tokenize(unicodes);
  std::string newtext;
  utf8::utf16to8(
      reinterpret_cast<const uint16_t*>(unicodes.c_str()),
      reinterpret_cast<const uint16_t*>(unicodes.c_str() + unicodes.size()),
      std::back_inserter(newtext));
  std::vector<std::string> tokens = absl::StrSplit(newtext, ' ');
  for (auto s : tokens) {
    if (s.size() > kMaxCharsPerWords) {
      results.push_back(token_2_id_map_.at(kUnkToken));
    } else {
      max_seg_(s, results);
    }
  }
  return results;
}

int BertTokenizer::PadId() const { return token_2_id_map_.at(kPadToken); }
int BertTokenizer::MaskId() const { return token_2_id_map_.at(kMaskToken); }
int BertTokenizer::SepId() const { return token_2_id_map_.at(kSepToken); }
int BertTokenizer::ClsId() const { return token_2_id_map_.at(kClsToken); }
int BertTokenizer::UnkId() const { return token_2_id_map_.at(kUnkToken); }

int BertTokenizer::TotalSize() const { return tokens_.size(); }
void BertTokenizer::max_seg_(std::string s, std::vector<int>& results) {
  int end = s.size();
  int start = 0;
  // spdlog::info("now s:[{}]", s);
  bool firstOne = true;
  while (start < end) {
    std::string test(s.c_str() + start, end - start);
    if (!firstOne) {
      test = std::string("##") + test;
    }
    auto it = token_2_id_map_.find(test);
    if (it == token_2_id_map_.end()) {
      end -= 1;
    } else {
      // spdlog::info("now got :{}", test);
      results.push_back(it->second);
      start = end;
      end = s.size();
      firstOne = false;
    }
  }
  if (firstOne) {
    // not any one matched
    results.push_back(token_2_id_map_.at(kUnkToken));
  }
}
int BertTokenizer::Word2Id(std::string s) const {
  if (s.size() > kMaxCharsPerWords) {
    return token_2_id_map_.at(kUnkToken);
  }
  auto it = token_2_id_map_.find(s);
  if (it == token_2_id_map_.end()) {
    return token_2_id_map_.at(kUnkToken);
  } else {
    return it->second;
  }
}
std::string BertTokenizer::Id2Word(int id) const {
  if (id >= 0 && id < static_cast<int>(tokens_.size())) {
    return tokens_[id];
  }
  return kUnkToken;
}

void BertTokenizer::load_vocab_(std::string path) {
  FILE* fp = fopen(path.c_str(), "r");
  CHECK(fp != NULL) << "open file error:" << path;
  char line[4096] = {0};
  int idx = 0;
  while (fgets(line, sizeof(line) - 1, fp)) {
    int nn = strlen(line);
    while (nn && (line[nn - 1] == '\n' || line[nn - 1] == '\r')) {
      nn -= 1;
    }
    if (nn <= 0) {
      continue;
    }
    std::string token(line, nn);
    tokens_.push_back(token);
    token_2_id_map_[token] = idx;
    idx += 1;
  }
  fclose(fp);
}

static bool _is_whitespace(uint16_t c) {
  if (c == '\t' || c == '\n' || c == '\r' || c == ' ') {
    return true;
  }
  return std::iswspace(c);
}

static bool _is_control(uint16_t c) {
  if (c == '\t' || c == '\n' || c == '\r') {
    return false;
  }
  return std::iswcntrl(c);
}

static bool _is_chinese_char(uint16_t cp) {
  if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;
  }
  return false;
}
static bool _is_punct_char(uint16_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  if (cp == ' ') {
    return false;
  }
  if (kChinesePunts.find(cp) != kChinesePunts.end()) {
    return true;
  }
  return std::iswpunct(cp);
}
UString BertTokenizer::_basic_tokenize(UString text) {
  UString ret;
  size_t len = text.size();
  for (size_t i = 0; i < len; i++) {
    uint16_t c = text[i];
    if (_is_chinese_char(c) || _is_punct_char(c)) {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, ' ');
      }
      ret.append(1, c);
      ret.append(1, ' ');
    } else if (c == ' ') {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, c);
      }
    } else {
      ret.append(1, c);
    }
  }
  if (!ret.empty() && ret.back() == ' ') {
    ret.erase(ret.end() - 1);
  }
  return ret;
}
UString BertTokenizer::_clean(UString text) {
  size_t len = text.size();
  UString ret;
  for (size_t i = 0; i < len; i++) {
    uint16_t c = text[i];
    if (c == 0 || c == 0xFFFD || _is_control(c)) {
      continue;
    }
    if (_is_whitespace(c)) {
      ret.append(1, ' ');
    } else {
      ret.append(1, c);
    }
  }
  return ret;
}

UString BertTokenizer::_strip_accents(UString text) {
  size_t len = text.size();
  UString ret;
  for (size_t i = 0; i < len; i++) {
    uint16_t c = text[i];
    if (utf8proc_category(c) == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    ret.append(1, c);
  }
  return ret;
}

}  // namespace radish