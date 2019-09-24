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
#include "google/protobuf/util/json_util.h"
#include "leveldb/db.h"
#include "torch/nn/module.h"
#include "torch/torch.h"
#include "torch/types.h"
#include "train/data/example_parser.h"
#include "train/data/llb_example.h"
#include "train/proto/example.pb.h"
#include "utils/logging.h"

namespace radish {
namespace data {
static std::string kTotalCountKey = "_TOTAL_COUNT_";
static std::string kConfigMetaKey = "_CONF_METADATA_";

template <class Parser>
class LeveldbDataset
    : public torch::data::Dataset<LeveldbDataset<Parser>, LlbExample> {
 public:
  explicit LeveldbDataset(const std::string& path) {
    leveldb::DB* db = nullptr;
    leveldb::Options opt;
    CHECK(leveldb::DB::Open(opt, path, &db).ok()) << "Open  db error:" << path;
    db_.reset(db);
    parser_.reset(new Parser());
    leveldb::ReadOptions ropt;
    std::string rawData;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(kConfigMetaKey), &rawData);
    if (!st.ok()) {
      spdlog::warn("no config metadata for parser!");
    } else {
      Json::Value conf;
      Json::Reader reader;
      spdlog::info("got config: {}!", rawData);
      CHECK(reader.parse(rawData, conf))
          << "parse from metadata conf error:" << rawData;
      CHECK(parser_->Init(conf)) << "Init example parser error";
      spdlog::info("init data example parser success!");
    }
  }
  virtual ~LeveldbDataset() {}

  LlbExample get(size_t index) override {
    leveldb::ReadOptions ropt;
    std::string rawData;
    LlbExample ret;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(std::to_string(index+1)), &rawData);
    if (!st.ok()) {
      spdlog::warn("key not found:{}", index+1);
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
      spdlog::warn("no total count key::{}", kTotalCountKey);
      return torch::nullopt;
    } else {
      size_t count = 0;
      if (!absl::SimpleAtoi(rawData, &count)) {
        return torch::nullopt;
      }
      spdlog::info("total count :::{}", count);
      return {count};
    }
  }

 private:
  std::shared_ptr<leveldb::DB> db_;
  std::shared_ptr<ExampleParser> parser_;
};

}  // namespace data
}  // namespace radish
