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
#include <random>
#include <string>

#include "absl/strings/str_split.h"
#include "torch/torch.h"
#include "torch/types.h"
#include "train/data/example_parser.h"
#include "train/data/llb_example.h"
#include "utils/logging.h"
namespace radish {
namespace data {

class TxtFile {
 public:
  TxtFile(std::string path) : infile_(path), path_(path) {
    CHECK(infile_) << path;
  }
  ~TxtFile() { infile_.close(); }
  bool NextLine(std::string& line) { return bool(std::getline(infile_, line)); }

 private:
  std::ifstream infile_;
  std::string path_;
};
template <class Parser>
class TxtDataset : public torch::data::Dataset<TxtDataset<Parser>, LlbExample> {
 public:
  explicit TxtDataset(std::string pathstr, const Json::Value& parserConf)
      : gen_(std::random_device{}()) {
    parser_.reset(new Parser());
    CHECK(parser_->Init(parserConf));
    std::vector<std::string> pathList = absl::StrSplit(pathstr, ",");
    CHECK(pathList.size() < kMaxFiles)
        << "path number should be less than :" << kMaxFiles;
    total_ = 0;
    for (size_t i = 0; i < pathList.size(); i++) {
      TxtFile txt(pathList[i]);
      std::string line;
      while (txt.NextLine(line)) {
        total_ += 1;
      }
      file_lists_.push_back(std::make_shared<TxtFile>(pathList[i]));
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
      std::uniform_int_distribution<size_t> rng(0, read_inds_.size()-1);
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