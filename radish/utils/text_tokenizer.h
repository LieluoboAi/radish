/*
 * File: text_tokenizer.h
 * Project: utils
 * File Created: Sunday, 20th October 2019 9:53:58 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 10:05:56 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#pragma once

#include <cxxabi.h>

#include <map>
#include <string>
#include <vector>

namespace radish {
class TextTokenizer {
 public:
  TextTokenizer() = default;
  virtual ~TextTokenizer() = default;
  virtual bool Init(std::string vocab) = 0;
  virtual std::vector<int> Encode(std::string text) = 0;
  virtual int Word2Id(std::string word) const = 0;
  virtual std::string Id2Word(int id) const = 0;
  virtual int PadId() const { return 0; }
  virtual int MaskId() const = 0;
  virtual int SepId() const = 0;
  virtual int ClsId() const = 0;
  virtual int UnkId() const = 0;
  virtual int TotalSize()  const =0;
};

class TextTokenizerFactory {
 public:
  using TCreateMethod = TextTokenizer* (*)();

 public:
  TextTokenizerFactory() = delete;

  static bool Register(const std::string name, TCreateMethod funcCreate);

  static TextTokenizer* Create(const std::string& name);

 private:
  static std::map<std::string, TCreateMethod>* sMethods;
};

template <typename T>
class TextTokenizerRegisteeStub {
 public:
  static std::string factory_name() {
    int status = 0;
    return std::string(abi::__cxa_demangle(typeid(T).name(), 0, 0, &status));
  }

 protected:
  static bool s_bRegistered;
};

template <typename T>
bool TextTokenizerRegisteeStub<T>::s_bRegistered =
    TextTokenizerFactory::Register(TextTokenizerRegisteeStub<T>::factory_name(),
                                   []() {
                                     return dynamic_cast<TextTokenizer*>(new T);
                                   });

}  // namespace radish
