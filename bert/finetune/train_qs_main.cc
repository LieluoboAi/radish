/*
 * File: train_qs_main.cc
 * Project: finetune
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-05 10:16:46
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "utils/logging.h"

#include "bert/finetune/query_same_model.h"
#include "bert/finetune/query_same_parser.h"

#include "train/llb_trainer.h"
#include "train/progress_reporter.h"

ABSL_FLAG(std::string, train_data_path, "/data/query/LCQMC/train.txt",
          "the train data path");
ABSL_FLAG(std::string, test_data_path, "/data/query/LCQMC/dev.txt",
          "the test data path");
ABSL_FLAG(std::string, parser_conf_path, "bert/parser_conf.json",
          "the example parser conf path");
ABSL_FLAG(std::string, logdir, "logs", "the model log dir ");
ABSL_FLAG(std::string, pretrained_model_path, "",
          "the pretrained model (albert etc..)  path ");
ABSL_FLAG(std::string, pretrain_prefix_var_name, "transformer_encoder",
          "the pretrain prefix variable name");
ABSL_FLAG(int32_t, n_vocab, 32003, "The vocab number of input tokens");
ABSL_FLAG(int32_t, max_seq_len, 512, "seq len of input ");
ABSL_FLAG(double, dropout, 0.1, "dropout rate");
ABSL_FLAG(int32_t, d_word_vec, 128, "dimension of word vec ");
ABSL_FLAG(int32_t, batch_size, 100, "batch size of trainning steps ");
ABSL_FLAG(int64_t, max_test_num, 0, "max test examples allowed");
ABSL_FLAG(int32_t, eval_every, 1000,
          "every X steps , evaluate once for test loss");
ABSL_FLAG(float, learning_rate, 0.0001, "the learning rate ");
ABSL_FLAG(int32_t, warmup_steps, 3000, "the warmup steps");

int main(int argc, char* argv[]) {
  // Passing params by value does NOT work correctly.
  absl::ParseCommandLine(argc, argv);
  radish::QuerySameModel model = radish::QuerySameModel(
      radish::QuerySameOptions(absl::GetFlag(FLAGS_n_vocab))
          .len_max_seq(absl::GetFlag(FLAGS_max_seq_len))
          .dropout(absl::GetFlag(FLAGS_dropout))
          .d_word_vec(absl::GetFlag(FLAGS_d_word_vec)));
  std::string logdir = absl::GetFlag(FLAGS_logdir);
  CHECK(!logdir.empty()) << "logdir should not be empty";
  std::string parserConfPath = absl::GetFlag(FLAGS_parser_conf_path);
  radish::train::ProgressReporter reporter;
  radish::train::LlbTrainer<radish::QSExampleParser, radish::QuerySameModel,
                            false, 10, true>
      trainner(logdir);
  std::string trainDataPath = absl::GetFlag(FLAGS_train_data_path);
  std::string testDataPath = absl::GetFlag(FLAGS_test_data_path);
  CHECK(!trainDataPath.empty()) << "train data path is empty";
  CHECK(!testDataPath.empty()) << "test data path is empty";
  trainner.MainLoop(
      model, trainDataPath, testDataPath, absl::GetFlag(FLAGS_learning_rate),
      absl::GetFlag(FLAGS_batch_size), absl::GetFlag(FLAGS_eval_every),
      &reporter, parserConfPath, 100 /** epoch */,
      absl::GetFlag(FLAGS_warmup_steps), absl::GetFlag(FLAGS_max_test_num),
      2 /** update per batchs */,
      absl::GetFlag(FLAGS_pretrained_model_path) /**pretrained model path*/,
      absl::GetFlag(
          FLAGS_pretrain_prefix_var_name) /**pretrain prefix var name*/
  );
  return 0;
}
