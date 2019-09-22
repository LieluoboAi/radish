/*
 * File: progress_reporter.h
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-18 10:05:57
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once
#include <stdint.h>
#include <string>
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "glog/logging.h"

namespace radish {
namespace train {
class ProgressReporter {
 public:
  virtual void UpdateProgress(int64_t step, absl::optional<double> trainLoss,
                              absl::optional<double> testLoss,
                              absl::optional<double> testEval) {
    std::string toLog;
    if (trainLoss != absl::nullopt) {
      absl::StrAppend(&toLog, "Step:", step,
                      "trainning loss:", trainLoss.value());
    }
    if (testLoss != absl::nullopt) {
      absl::StrAppend(&toLog, "test loss:", testLoss.value());
    }
    if (testEval != absl::nullopt) {
      absl::StrAppend(&toLog, "test eval:", testEval.value());
    }
  }
};

}  // namespace train
}  // namespace radish
