/*
 * File: text_tokenizer.cc
 * Project: utils
 * File Created: Sunday, 20th October 2019 10:33:18 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 10:34:00 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include "radish/utils/text_tokenizer.h"
#include "radish/utils/logging.h"
namespace radish {

std::map<std::string, TextTokenizerFactory::TCreateMethod>*
    TextTokenizerFactory::sMethods =
        nullptr;  // Use pointer to walk around wild init issue!!!!

bool TextTokenizerFactory::Register(
    const std::string name, TextTokenizerFactory::TCreateMethod funcCreate) {
  if (sMethods == nullptr) {
    sMethods = new std::map<std::string, TextTokenizerFactory::TCreateMethod>();
  }
  auto it = sMethods->find(name);
  if (it == sMethods->end()) {
    sMethods->insert(std::make_pair(name, funcCreate));
    return true;
  }
  return false;
}

TextTokenizer* TextTokenizerFactory::Create(const std::string& name) {
  auto it = sMethods->find(name);
  if (it == sMethods->end()) {
    return nullptr;
  }
  return it->second();
}
}  // namespace radish
