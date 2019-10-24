/*
 * File: benchmark_submiter.h
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-23 3:23:28
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once
#include <stdint.h>
#include <fstream>
#include <string>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "radish/utils/logging.h"

namespace radish {
namespace train {
class BenchmarkSubmiter {
 public:
  virtual void SubmitOneRow(std::vector<float> output) = 0;
  virtual void SubmitDone() {}
};

class FileBenchmarkSubmiter : public BenchmarkSubmiter {
 public:
  FileBenchmarkSubmiter(const std::string outputPath) : out_(outputPath) {}
  void SubmitOneRow(std::vector<float> outputs) override {
    std::string line;
    if (!outputs.empty()) {
      absl::StrAppend(&line, outputs[0]);
      for (size_t i = 1; i < outputs.size(); i++) {
        absl::StrAppend(&line, "\t", outputs[i]);
      }
    }
    out_ << line << std::endl;
  }
  void SubmitDone() { out_.close(); }

 private:
  std::ofstream out_;
};
}  // namespace train
}  // namespace radish
