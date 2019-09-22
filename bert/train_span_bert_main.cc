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
#include "glog/logging.h"

#include "bert/span_bert_example_parser.h"
#include "bert/span_bert_model.h"

#include "train/llb_trainer.h"
#include "train/progress_reporter.h"

ABSL_FLAG(std::string, train_data_path, "data/train", "the train data path");
ABSL_FLAG(std::string, test_data_path, "data/test", "the test data path");
ABSL_FLAG(int32_t, n_vocab, 32003, "The vocab number of input tokens");
ABSL_FLAG(int32_t, max_seq_len, 200, "seq len of input ");
ABSL_FLAG(int32_t, d_word_vec, 200, "dimension of word vec ");
ABSL_FLAG(int32_t, batch_size, 64, "batch size of trainning steps ");
ABSL_FLAG(int32_t, eval_every, 5000,
          "every X steps , evaluate once for test loss");
ABSL_FLAG(float, learning_rate, 0.0001, "the learning rate ");
ABSL_FLAG(int32_t, warmup_steps, 60000, "the warmup steps");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  knlp::SpanBertModel model = knlp::SpanBertModel(
      absl::GetFlag(FLAGS_n_vocab), absl::GetFlag(FLAGS_max_seq_len),
      absl::GetFlag(FLAGS_d_word_vec));
  knlp::train::ProgressReporter reporter;
  knlp::train::LlbTrainer<knlp::SpanBertExampleParser, knlp::SpanBertModel>
      trainner;
  std::string trainDataPath = absl::GetFlag(FLAGS_train_data_path);
  std::string testDataPath = absl::GetFlag(FLAGS_test_data_path);
  CHECK(!trainDataPath.empty()) << "train data path is empty";
  CHECK(!testDataPath.empty()) << "test data path is empty";
  trainner.MainLoop(
      model, trainDataPath, testDataPath, absl::GetFlag(FLAGS_learning_rate),
      absl::GetFlag(FLAGS_batch_size), absl::GetFlag(FLAGS_eval_every),
      &reporter, absl::GetFlag(FLAGS_warmup_steps));
  return 0;
}
