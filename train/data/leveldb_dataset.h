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

#include <torch/nn/module.h>
#include <torch/torch.h>

#include <torch/types.h>
#include <string>
#include "absl/strings/numbers.h"
#include "glog/logging.h"
#include "google/protobuf/util/json_util.h"
#include "leveldb/db.h"
#include "train/proto/example.pb.h"

namespace knlp {
namespace data {
static std::string kTotalCountKey = "_TOTAL_COUNT_";
template <class SampleT>
class LeveldbDataset
    : public torch::data::Dataset<LeveldbDataset<SampleT>, SampleT> {
 public:
  LeveldbDataset(const std::string& path) {
    leveldb::Options opt;
    CHECK(!leveldb::DB::Open(opt, path, &db_).ok())
        << "Open tagged db error:" << path;
  }
  virtual ~LeveldbDataset() {
    if (db_) {
      delete db_;
    }
    db_ = nullptr;
  }

  SampleT get(size_t index) override {
    leveldb::ReadOptions ropt;
    std::string rawData;
    SampleT ret;
    leveldb::Status st =
        db_->Get(ropt, leveldb::Slice(std::to_string(index)), &rawData);
    if (!st.ok()) {
      VLOG(0) << "key not found:" << index;
      return ret;
    }
    train::TrainExample exampleProto;
    exampleProto.ParseFromString(rawData);
    ret.FromMessage(exampleProto);
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
};

}  // namespace data
}  // namespace knlp
