/*
 * File: txt_dataset.h
 * Project: data
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-27 9:57:24
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <stdlib.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>

#include "absl/strings/str_split.h"
#include "radish/train/data/example_parser.h"
#include "radish/train/data/llb_example.h"
#include "radish/utils/logging.h"
#include "torch/torch.h"
#include "torch/types.h"
namespace radish {
namespace data {

class TxtFile {
 public:
  TxtFile(std::string path, int hint = 0)
      : infile_(path), path_(path), hint_(hint), done_(false) {
    CHECK(infile_) << path;
    if (absl::EndsWith(path, ".tsv") || absl::EndsWith(path, ".csv")) {
      // make sure csv has head??????
      std::string line;
      CHECK(std::getline(infile_, line));
    }
  }
  ~TxtFile() { infile_.close(); }
  bool NextLine(std::string& line) {
    std::lock_guard<std::mutex> _(lock_);
    if (hint_ > 0) {
      if (preload_buffers_.empty()) {
        if (done_) {
          return false;
        }
        if (!pre_load_()) {
          return false;
        }
      }
      line = preload_buffers_.back();
      preload_buffers_.erase(preload_buffers_.begin() +
                             preload_buffers_.size() - 1);
      return true;
    } else {
      return bool(std::getline(infile_, line));
    }
  }

 private:
  bool pre_load_() {
    if (done_) {
      return false;
    }
    for (int i = 0; i < hint_; i++) {
      std::string str;
      if (std::getline(infile_, str)) {
        preload_buffers_.push_back(str);
      } else {
        done_ = true;
        break;
      }
    }
    // spdlog::info("preloaded {} exs.", preload_buffers_.size());
    std::shuffle(preload_buffers_.begin(), preload_buffers_.end(),
                 std::mt19937());
    return !preload_buffers_.empty();
  }
  std::ifstream infile_;
  std::string path_;
  std::mutex lock_;
  int hint_;
  std::vector<std::string> preload_buffers_;
  bool done_;
};
template <class Parser>
class TxtDataset : public torch::data::Dataset<TxtDataset<Parser>, LlbExample> {
 public:
  explicit TxtDataset(std::string pathstr, const Json::Value& parserConf)
      : gen_(std::random_device{}()) {
    parser_.reset(new Parser());
    CHECK(parser_->Init(parserConf));
    int preload = parserConf.get("parser.preload", 1000).asInt();
    if (preload == 0) {
      spdlog::info("manually disabled preload!");
    }
    std::vector<std::string> pathList = absl::StrSplit(pathstr, ",");
    int hint = preload / pathList.size();
    CHECK(pathList.size() < kMaxFiles)
        << "path number should be less than :" << kMaxFiles;
    total_ = 0;
    for (size_t i = 0; i < pathList.size(); i++) {
      TxtFile txt(pathList[i]);
      std::string line;
      while (txt.NextLine(line)) {
        total_ += 1;
      }
      file_lists_.push_back(std::make_shared<TxtFile>(pathList[i], hint));
      read_inds_.push_back(i);
    }
    spdlog::info("total {} records", total_);
  }
  virtual ~TxtDataset() {}

  LlbExample get(size_t index) override {
    size_t idx = 0;
    LlbExample ret;
    bool gotIdx = false;
    std::string rawData;
    while (!gotIdx && !read_inds_.empty()) {
      std::uniform_int_distribution<size_t> rng(0, read_inds_.size() - 1);
      idx = rng(gen_);
      if (file_lists_[idx]->NextLine(rawData)) {
        gotIdx = true;
        if (!parser_->ParseOne(rawData, ret)) {
          spdlog::warn("Parser example error");
          ret.features.clear();
        }
      } else {
        // 该文件读完了
        read_inds_.erase(read_inds_.begin() + idx);
      }
    }
    return ret;
  }
  torch::optional<size_t> size() const override { return {total_}; }

 private:
  std::shared_ptr<ExampleParser> parser_;
  std::vector<std::shared_ptr<TxtFile>> file_lists_;
  // 读头
  std::vector<size_t> read_inds_;
  size_t total_;
  std::mt19937 gen_;
  static const size_t kMaxFiles = 128;
};

}  // namespace data
}  // namespace radish
