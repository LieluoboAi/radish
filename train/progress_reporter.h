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
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "utils/logging.h"

namespace radish {
namespace train {
class ProgressReporter {
 public:
  virtual void UpdateProgress(int64_t step, absl::optional<float> trainLoss,
                              absl::optional<float> testLoss,
                              absl::optional<std::vector<float>> evals) {
    if (step % 100) {
      return;
    }
    std::string toLog;
    if (trainLoss != absl::nullopt) {
      absl::StrAppend(&toLog, "Step:", step,
                      "   trainning loss:", trainLoss.value());
    }
    if (testLoss != absl::nullopt) {
      absl::StrAppend(&toLog, "  test loss:", testLoss.value());
    }
    if (evals != absl::nullopt) {
      absl::StrAppend(&toLog,
                      "  test evals:", absl::StrJoin(evals.value(), "|"));
    }
    spdlog::info(toLog);
  }
};

}  // namespace train
}  // namespace radish
