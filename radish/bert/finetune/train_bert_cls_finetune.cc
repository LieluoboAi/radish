/*
 * File: train_bert_cls_finetune.cc
 * Project: finetune
 * File Created: Sunday, 20th October 2019 2:30:33 pm
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 2:30:35 pm
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "radish/utils/logging.h"

#include "radish/bert/finetune/bert_classification_model.h"
#include "radish/bert/finetune/xnli_example_parser.h"
#include "radish/train/llb_trainer.h"
#include "radish/train/progress_reporter.h"

ABSL_FLAG(std::string, train_data_path, "/e/data/chineseGLUE/xnli/train.tsv",
          "the train data path");
ABSL_FLAG(std::string, test_data_path, "/e/data/chineseGLUE/xnli/test.tsv",
          "the test data path");
ABSL_FLAG(std::string, parser_conf_path, "radish/bert/parser_conf.json",
          "the example parser conf path");
ABSL_FLAG(std::string, logdir, "logs", "the model log dir ");
ABSL_FLAG(std::string, pretrained_model_path, "",
          "the pretrained model  path ");
ABSL_FLAG(int32_t, batch_size, 64, "batch size of trainning steps ");
ABSL_FLAG(int64_t, max_test_num, 0, "max test examples allowed");
ABSL_FLAG(int32_t, eval_every, 1000,
          "every X steps , evaluate once for test loss");
ABSL_FLAG(float, learning_rate, 0.0001, "the learning rate ");
ABSL_FLAG(int32_t, warmup_steps, 1000, "the warmup steps");
int main(int argc, char* argv[]) {
  // Passing params by value does NOT work correctly.
  absl::ParseCommandLine(argc, argv);

  radish::BertClassificationModel model =
      radish::BertClassificationModel(radish::BertOptions::kBertBaseOpts, 3);
  std::string logdir = absl::GetFlag(FLAGS_logdir);
  CHECK(!logdir.empty()) << "logdir should not be empty";
  std::string parserConfPath = absl::GetFlag(FLAGS_parser_conf_path);
  radish::train::ProgressReporter reporter;

  std::string trainDataPath = absl::GetFlag(FLAGS_train_data_path);
  std::string testDataPath = absl::GetFlag(FLAGS_test_data_path);
  CHECK(!trainDataPath.empty()) << "train data path is empty";
  CHECK(!testDataPath.empty()) << "test data path is empty";

  radish::train::LlbTrainer<radish::XNLIExampleParser,
                            radish::BertClassificationModel, false, 10, true>
      trainner(logdir);
  trainner.MainLoop(
      model, trainDataPath, testDataPath, absl::GetFlag(FLAGS_learning_rate),
      absl::GetFlag(FLAGS_batch_size), absl::GetFlag(FLAGS_eval_every),
      &reporter, parserConfPath, 100 /** epoch */,
      absl::GetFlag(FLAGS_warmup_steps), absl::GetFlag(FLAGS_max_test_num),
      1 /** update per batchs */,
      absl::GetFlag(FLAGS_pretrained_model_path) /**pretrained model path*/
  );

  return 0;
}
