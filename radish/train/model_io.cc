/*
 * File: model_io.cc
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-24 4:09:12
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "radish/train/model_io.h"

#include <iostream>
#include <regex>
#include <stack>

#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/reader.h"

#include "radish/utils/logging.h"
#include "radish/utils/tensor_util.h"

namespace radish {
namespace train {

void SaveModel(std::shared_ptr<torch::nn::Module> module,
               const std::string& file_name) {
  torch::serialize::OutputArchive archive;
  auto params = module->named_parameters(true /*recurse*/);
  auto buffers = module->named_buffers(true /*recurse*/);
  for (const auto& val : params) {
    if (!radish::utils::IsEmpty(val.value())) {
      archive.write(val.key(), val.value());
    }
  }
  for (const auto& val : buffers) {
    if (!radish::utils::IsEmpty(val.value())) {
      archive.write(val.key(), val.value(), /*is_buffer*/ true);
    }
  }
  archive.save_to(file_name);
}

void LoadModel(std::shared_ptr<torch::nn::Module> module,
               const std::string& path, const std::string& ignore_name_regex,
               torch::Device device) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  torch::NoGradGuard no_grad;
  std::regex re(ignore_name_regex);
  std::smatch m;
  auto params = module->named_parameters(true /*recurse*/);
  auto buffers = module->named_buffers(true /*recurse*/);
  for (auto& val : params) {
    if (!std::regex_match(val.key(), m, re)) {
      archive.read(val.key(), val.value());
    }
  }
  for (auto& val : buffers) {
    if (!std::regex_match(val.key(), m, re)) {
      archive.read(val.key(), val.value(), /*is_buffer*/ true);
    }
  }
}

void LoadModelEx(std::shared_ptr<torch::nn::Module> module,
                 const std::string& path, const std::string& prefixVarName,
                 torch::Device device) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  torch::NoGradGuard no_grad;
  auto params = module->named_parameters(true /*recurse*/);
  auto buffers = module->named_buffers(true /*recurse*/);
  for (auto& val : params) {
    std::string kn = val.key();
    if (prefixVarName.empty() || (kn.size() >= prefixVarName.size() &&
                                  strncmp(kn.c_str(), prefixVarName.c_str(),
                                          prefixVarName.size()) == 0)) {
      archive.read(kn, val.value());
      spdlog::info("load pretrained weights:{}", kn);
    }
  }
  for (auto& val : buffers) {
    std::string kn = val.key();
    if (prefixVarName.empty() || (kn.size() >= prefixVarName.size() &&
                                  strncmp(kn.c_str(), prefixVarName.c_str(),
                                          prefixVarName.size()) == 0)) {
      archive.read(val.key(), val.value(), /*is_buffer*/ true);
      spdlog::info("load pretrained buffer:{}", kn);
    }
  }
}

namespace {

enum class ReadState {
  None,
  DictObject,
  ParamName,
  SizeTensorPair,
  TensorSize,
  SizeTensorPairDelim,
  TensorValue,
  List
};

struct DictHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DictHandler> {
  DictHandler() {}

  bool Double(double d) {
    if (current_state_.top() == ReadState::List ||
        current_state_.top() == ReadState::TensorValue) {
      blob_.push_back(static_cast<float>(d));
      ++index_;
    } else {
      throw std::logic_error("Double parsing error");
    }
    return true;
  }

  bool Uint(unsigned u) {
    if (current_state_.top() == ReadState::List ||
        current_state_.top() == ReadState::TensorValue) {
      blob_.push_back(static_cast<float>(u));
      ++index_;
    } else if (current_state_.top() == ReadState::TensorSize) {
      size_.push_back(static_cast<int64_t>(u));
    } else {
      throw std::logic_error("UInt parsing error");
    }
    return true;
  }

  bool Key(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    key_.assign(str, length);
    if (current_state_.top() == ReadState::DictObject) {
      current_state_.push(ReadState::ParamName);
    } else {
      throw std::logic_error("Key parsing error");
    }
    return true;
  }

  bool StartObject() {
    if (current_state_.top() == ReadState::None) {
      current_state_.pop();
      current_state_.push(ReadState::DictObject);
    } else {
      throw std::logic_error("Start object parsing error");
    }
    return true;
  }

  bool EndObject(rapidjson::SizeType /*memberCount*/) {
    if (current_state_.top() != ReadState::DictObject) {
      throw std::logic_error("End object parsing error");
    }
    return true;
  }

  void StartData() {
    current_state_.push(ReadState::TensorValue);
    auto total_length = std::accumulate(size_.begin(), size_.end(), 1,
                                        std::multiplies<int64_t>());
    blob_.resize(static_cast<size_t>(total_length));
    blob_.clear();
    index_ = 0;
  }

  bool StartArray() {
    if (current_state_.top() == ReadState::List) {
      current_state_.push(ReadState::List);
    } else if (current_state_.top() == ReadState::ParamName) {
      current_state_.push(ReadState::SizeTensorPair);
    } else if (current_state_.top() == ReadState::SizeTensorPair) {
      current_state_.push(ReadState::TensorSize);
      size_.clear();
    } else if (current_state_.top() == ReadState::SizeTensorPairDelim) {
      current_state_.pop();
      StartData();
    } else if (current_state_.top() == ReadState::TensorValue) {
      current_state_.push(ReadState::List);
    } else {
      throw std::logic_error("Start array parsing error");
    }
    return true;
  }

  bool EndArray(rapidjson::SizeType elementCount) {
    if (current_state_.top() == ReadState::List) {
      current_state_.pop();
    } else if (current_state_.top() == ReadState::SizeTensorPair) {
      current_state_.pop();
      assert(current_state_.top() == ReadState::ParamName);
      current_state_.pop();
      dict.insert(key_, tensor_);
      spdlog::info("{} : {},{}", key_, tensor_.type().toString(),
                   tensor_.dim());
    } else if (current_state_.top() == ReadState::TensorSize) {
      current_state_.pop();
      if (elementCount == 0) {
        size_.push_back(1);
        StartData();
      } else {
        current_state_.push(ReadState::SizeTensorPairDelim);
      }
    } else if (current_state_.top() == ReadState::TensorValue) {
      current_state_.pop();
      assert(index_ == static_cast<int64_t>(blob_.size()));
      tensor_ = torch::from_blob(blob_.data(), at::IntList(size_),
                                 at::dtype(at::kFloat))
                    .clone();  // clone to copy temp data
      if (blob_.size() == 1) {
        assert(current_state_.top() == ReadState::SizeTensorPair);
        current_state_.pop();
        assert(current_state_.top() == ReadState::ParamName);
        current_state_.pop();
        dict.insert(key_, tensor_);
        spdlog::info("{} : {},{}", key_, tensor_.type().toString(),
                     tensor_.dim());
      }
    } else {
      throw std::logic_error("End array parsing error");
    }
    return true;
  }

  std::string key_;
  std::vector<int64_t> size_;
  torch::Tensor tensor_;
  std::vector<float> blob_;
  int64_t index_{0};

  std::stack<ReadState> current_state_{{ReadState::None}};

  torch::OrderedDict<std::string, torch::Tensor> dict;
};
}  // namespace

torch::OrderedDict<std::string, torch::Tensor> LoadStateDictJson(
    const std::string& file_name) {
  auto* file = std::fopen(file_name.c_str(), "r");
  if (file) {
    char readBuffer[65536];
    rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
    rapidjson::Reader reader;
    DictHandler handler;
    auto res = reader.Parse(is, handler);
    std::fclose(file);

    if (!res) {
      throw std::runtime_error(rapidjson::GetParseError_En(res.Code()));
    }

    return handler.dict;
  }
  return torch::OrderedDict<std::string, torch::Tensor>();
}

void LoadStateDictJson(std::shared_ptr<torch::nn::Module> module,
                       const std::string& file_name) {
  if (file_name.find(".json") != std::string::npos) {
    torch::NoGradGuard no_grad;
    auto new_params = LoadStateDictJson(file_name);
    auto params = module->named_parameters(true /*recurse*/);
    auto buffers = module->named_buffers(true /*recurse*/);

    for (auto& val : new_params) {
      auto name = val.key();
      // fix naming
      auto pos = name.find("running_var");
      if (pos != std::string::npos) {
        name.replace(pos, 11, "running_variance");
      }

      auto* t = params.find(name);
      if (t != nullptr) {
        spdlog::info("{}   copy, shape {}", name, val.value().sizes());
        t->copy_(val.value());
      } else {
        t = buffers.find(name);
        if (t != nullptr) {
          spdlog::info("{}   copy buffer , shape {}", name,
                       val.value().sizes());
          t->copy_(val.value());
        } else {
          spdlog::info("{}   parameter not found!", name);
        }
      }
    }

    auto pos = file_name.find_last_of(".");
    std::string new_file_name = file_name.substr(0, pos + 1);
    new_file_name += "dat";
    SaveModel(module, new_file_name);
    spdlog::info("Model state converted to file :{}", new_file_name);
  } else {
    throw std::invalid_argument("Can't load not a Json file!");
  }
}

}  // namespace train
}  // namespace radish
