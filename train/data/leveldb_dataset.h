/*
 * File: leveldb_dataset.h
 * Project: data
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-18 2:56:09
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <memory>
#include <string>

#include "absl/strings/numbers.h"
#include "glog/logging.h"
#include "google/protobuf/util/json_util.h"
#include "leveldb/db.h"
#include "torch/nn/module.h"
#include "torch/torch.h"
#include "torch/types.h"
#include "train/data/example_parser.h"
#include "train/data/llb_example.h"
#include "train/proto/example.pb.h"

namespace radish {
namespace data {
static std::string kTotalCountKey = "_TOTAL_COUNT_";
static std::string kConfigMetaKey = "_CONF_METADATA_";

template <class Parser>
class LeveldbDataset
    : public torch::data::Dataset<LeveldbDataset<Parser>, LlbExample> {
 public:
  explicit LeveldbDataset(const std::string& path) {
    leveldb::Options opt;
    CHECK(!leveldb::DB::Open(opt, path, &db_).ok())
        << "Open tagged db error:" << path;
    parser_.reset(new Parser());
    leveldb::ReadOptions ropt;
    std::string rawData;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(kConfigMetaKey), &rawData);
    if (!st.ok()) {
      VLOG(0) << "no config metadata for parser!";
    } else {
      Json::Value conf;
      Json::Reader reader;
      CHECK(reader.parse(rawData, conf))
          << "parse from metadata conf error:" << rawData;
      CHECK(parser_->Init(conf)) << "Init example parser error";
    }
  }
  virtual ~LeveldbDataset() {
    if (db_) {
      delete db_;
    }
    db_ = nullptr;
  }

  LlbExample get(size_t index) override {
    leveldb::ReadOptions ropt;
    std::string rawData;
    LlbExample ret;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(std::to_string(index)), &rawData);
    if (!st.ok()) {
      VLOG(0) << "key not found:" << index;
      return ret;
    }
    radish::train::TrainExample exampleProto;
    exampleProto.ParseFromString(rawData);
    CHECK(parser_->ParseOne(exampleProto, ret)) << "Parser example error";
    return ret;
  }
  torch::optional<size_t> size() const override {
    leveldb::ReadOptions ropt;
    std::string rawData;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(kTotalCountKey), &rawData);
    if (!st.ok()) {
      VLOG(0) << "no total count key:" << kTotalCountKey;
      return torch::nullopt;
    } else {
      size_t count = 0;
      if (!absl::SimpleAtoi(rawData, &count)) {
        return torch::nullopt;
      }
      return {count};
    }
  }

 private:
  leveldb::DB* db_ = nullptr;
  std::shared_ptr<ExampleParser> parser_;
};

}  // namespace data
}  // namespace radish
