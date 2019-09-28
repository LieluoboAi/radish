/**
 * @ Author: Koth
 * @ Create Time: 2019-09-28 12:45:56
 * @ Modified by: Koth
 * @ Modified time: 2019-09-28 13:52:11
 * @ Description:
 */

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "utils/logging.h"

#include "bert/albert_example_parser.h"
#include "bert/albert_model.h"

#include "train/llb_trainer.h"
#include "train/progress_reporter.h"

ABSL_FLAG(std::string, train_data_path,
          "/e/data/albert_data/albert/part0,/e/data/albert_data/albert/part1,/"
          "e/data/albert_data/albert/part2,/e/data/albert_data/albert/part3,/"
          "e/data/albert_data/albert/part4,/e/data/albert_data/albert/part5,/e/"
          "data/albert_data/albert/part6,/e/data/albert_data/albert/part7,/e/"
          "data/albert_data/albert/part8",
          "the train data path");
ABSL_FLAG(std::string, test_data_path, "/e/data/albert_data/albert/valid0",
          "the test data path");
ABSL_FLAG(std::string, parser_conf_path, "bert/parser_conf.json",
          "the example parser conf path");
ABSL_FLAG(std::string, logdir, "logs", "the model log dir ");
ABSL_FLAG(int32_t, n_vocab, 32003, "The vocab number of input tokens");
ABSL_FLAG(int32_t, max_seq_len, 512, "seq len of input ");
ABSL_FLAG(int32_t, d_word_vec, 128, "dimension of word vec ");
ABSL_FLAG(int32_t, batch_size, 512, "batch size of trainning steps ");
ABSL_FLAG(int64_t, max_test_num, 0, "max test examples allowed");
ABSL_FLAG(int32_t, eval_every, 6000,
          "every X steps , evaluate once for test loss");
ABSL_FLAG(float, learning_rate, 0.0001, "the learning rate ");
ABSL_FLAG(int32_t, warmup_steps, 40000, "the warmup steps");

int main(int argc, char* argv[]) {
  // Passing params by value does NOT work correctly.
  absl::ParseCommandLine(argc, argv);
  radish::ALBertModel model =
      radish::ALBertModel(radish::ALBertOptions(absl::GetFlag(FLAGS_n_vocab))
                              .len_max_seq(absl::GetFlag(FLAGS_max_seq_len))
                              .d_word_vec(absl::GetFlag(FLAGS_d_word_vec)));
  std::string logdir = absl::GetFlag(FLAGS_logdir);
  CHECK(!logdir.empty()) << "logdir should not be empty";
  std::string parserConfPath = absl::GetFlag(FLAGS_parser_conf_path);
  radish::train::ProgressReporter reporter;
  radish::train::LlbTrainer<radish::ALBertExampleParser, radish::ALBertModel,
                            false, 8, true>
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
      1 /** update per batchs */);
  return 0;
}
