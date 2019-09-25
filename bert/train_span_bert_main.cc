/*
 * File: train_span_bert_main.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-20 10:57:46
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "utils/logging.h"

#include "bert/span_bert_example_parser.h"
#include "bert/span_bert_model.h"

#include "train/llb_trainer.h"
#include "train/progress_reporter.h"

ABSL_FLAG(std::string, train_data_path, "spanbert_leveldb/valid",
          "the train data path");
ABSL_FLAG(std::string, test_data_path, "spanbert_leveldb/test",
          "the test data path");
ABSL_FLAG(std::string, logdir, "logs", "the model log dir ");
ABSL_FLAG(int32_t, n_vocab, 32003, "The vocab number of input tokens");
ABSL_FLAG(int32_t, max_seq_len, 512, "seq len of input ");
ABSL_FLAG(int32_t, d_word_vec, 200, "dimension of word vec ");
ABSL_FLAG(int32_t, batch_size, 400, "batch size of trainning steps ");
ABSL_FLAG(int64_t, max_test_num, 0, "max test examples allowed");
ABSL_FLAG(int32_t, eval_every, 6000,
          "every X steps , evaluate once for test loss");
ABSL_FLAG(float, learning_rate, 0.0001, "the learning rate ");
ABSL_FLAG(int32_t, warmup_steps, 40000, "the warmup steps");

int main(int argc, char* argv[]) {
  // Passing params by value does NOT work correctly.
  absl::ParseCommandLine(argc, argv);
  radish::SpanBertModel model = radish::SpanBertModel(
      absl::GetFlag(FLAGS_n_vocab), absl::GetFlag(FLAGS_max_seq_len),
      absl::GetFlag(FLAGS_d_word_vec));
  std::string logdir = absl::GetFlag(FLAGS_logdir);
  CHECK(!logdir.empty()) << "logdir should not be empty";
  radish::train::ProgressReporter reporter;
  radish::train::LlbTrainer<radish::SpanBertExampleParser,
                            radish::SpanBertModel, false, 8, 2>
      trainner(logdir);
  std::string trainDataPath = absl::GetFlag(FLAGS_train_data_path);
  std::string testDataPath = absl::GetFlag(FLAGS_test_data_path);
  CHECK(!trainDataPath.empty()) << "train data path is empty";
  CHECK(!testDataPath.empty()) << "test data path is empty";
  trainner.MainLoop(
      model, trainDataPath, testDataPath, absl::GetFlag(FLAGS_learning_rate),
      absl::GetFlag(FLAGS_batch_size), absl::GetFlag(FLAGS_eval_every),
      &reporter, 100 /** epoch */, absl::GetFlag(FLAGS_warmup_steps),
      absl::GetFlag(FLAGS_max_test_num));
  return 0;
}
