/*
 * File: prepare_leveldb_dataset.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-23 1:55:57
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"

#include "json/json.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "radish/train/proto/example.pb.h"

ABSL_FLAG(std::string, output_path, "data/train", "the output leveldb path");
ABSL_FLAG(int32_t, end_part, 9, "end part id");
ABSL_FLAG(int32_t, start_part, 1, "start part id");
ABSL_FLAG(std::string, input_path_prefix, "/data/chenyw/work_corpus/part-r",
          "the test data path");
ABSL_FLAG(std::string, spp_model_path, "char_model_32k.model",
          "the sentencepiece model path");

static int kBufferSize = 1024 * 4096;

class IdUnique {
 public:
  int GetNextId() {
    std::lock_guard<std::mutex> _(mutex_);
    id_ += 1;
    if ((id_ % 10000) == 0) {
      VLOG(0) << "generated :" << id_;
    }
    return id_;
  }
  int GetId() const { return id_; }

 private:
  int id_ = 0;
  std::mutex mutex_;
};
class DataWorker {
 public:
  DataWorker(leveldb::DB* db, IdUnique* idUnique) : db_(db), idu_(idUnique) {}
  void DoWork(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "r");
    if (fp == NULL) {
      VLOG(0) << "open file error:" << path;
      return;
    }
    Json::Reader reader;
    char* lineBuffer = new char[kBufferSize];
    lineBuffer[0] = '\0';
    int added = 0;
    std::vector<std::pair<std::string, std::string>> examples;
    while (fgets(lineBuffer, kBufferSize - 1, fp)) {
      int nn = strlen(lineBuffer);
      std::string str(lineBuffer, nn);
      Json::Value jv;
      if (!reader.parse(str, jv)) {
        VLOG(0) << "error parsing :\n" << str;
        delete[] lineBuffer;
        fclose(fp);
        return;
      }
      std::string desc = jv.get("des", "").asString();
      if (desc.size() < 20) {
        // VLOG(0) << "desc too short:" << desc;
        continue;
      }
      radish::train::TrainExample texample;
      texample.mutable_string_feature()->insert({"x", desc});
      int id = idu_->GetNextId();
      examples.push_back({std::to_string(id), texample.SerializeAsString()});
      if ((examples.size() % 100) == 0) {
        batch_add(examples);
        examples.clear();
      }
      added += 1;
    }
    if (!examples.empty()) {
      batch_add(examples);
    }
    VLOG(0) << "added :" << added;
    delete[] lineBuffer;
    fclose(fp);
  }

 private:
  void batch_add(
      const std::vector<std::pair<std::string, std::string>>& examples) {
    leveldb::WriteBatch batch;
    for (auto [key, value] : examples) {
      batch.Put(leveldb::Slice(key), leveldb::Slice(value));
    }
    leveldb::Status st = db_->Write(leveldb::WriteOptions(), &batch);
    if (!st.ok()) {
      CHECK(false) << "batch write error:" << st.ToString();
    }
  }
  leveldb::DB* db_;
  IdUnique* idu_;
};
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  IdUnique idUnique;
  std::vector<DataWorker*> workers;
  std::vector<std::thread> threads;
  int from = absl::GetFlag(FLAGS_start_part);
  int to = absl::GetFlag(FLAGS_end_part);
  leveldb::Options opt;
  opt.create_if_missing = true;
  opt.error_if_exists = true;
  opt.max_file_size = 1024 * 1024 * 256;
  opt.write_buffer_size = 1024 * 1024 * 64;
  leveldb::DB* db = nullptr;
  std::string outputPath = absl::GetFlag(FLAGS_output_path);
  CHECK(leveldb::DB::Open(opt, outputPath, &db).ok())
      << "Open  db error:" << outputPath;

  std::string spmPath = absl::GetFlag(FLAGS_spp_model_path);
  for (int i = from; i <= to; i++) {
    char path[256] = {0};
    int nl = snprintf(path, sizeof(path) - 1, "%s-%05d",
                      absl::GetFlag(FLAGS_input_path_prefix).c_str(), i);
    path[nl] = 0;
    workers.push_back(new DataWorker(db, &idUnique));
    threads.push_back(
        std::thread(&DataWorker::DoWork, workers.back(), std::string(path)));
  }
  for (int i = from; i <= to; i++) {
    threads[i - from].join();
  }
  leveldb::WriteOptions wo;
  Json::Value conf;
  conf["spm_model_path"] = absl::GetFlag(FLAGS_spp_model_path);
  Json::FastWriter writer;
  std::string confJson = writer.write(conf);
  CHECK(db->Put(wo, std::string("_CONF_METADATA_"), confJson).ok());
  CHECK(db->Put(wo, std::string("_TOTAL_COUNT_"),
                std::to_string(idUnique.GetId()))
            .ok());
  return 0;
}
