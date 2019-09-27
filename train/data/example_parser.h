/**
 * @ Author: Koth
 * @ Create Time: 2019-09-22 15:52:25
 * @ Modified by: Koth
 * @ Modified time: 2019-09-22 17:53:52
 * @ Description:
 */
#pragma once
#include <memory>
#include <string>

#include "json/json.h"
#include "torch/torch.h"
#include "train/data/llb_example.h"
#include "train/proto/example.pb.h"
namespace radish {
namespace data {
using Tensor = torch::Tensor;
class ExampleParser {
 public:
  virtual ~ExampleParser() {}
  virtual bool Init(const Json::Value& config) = 0;
  virtual bool ParseOne(train::TrainExample& protoData, LlbExample& example) {
    return false;
  }
  virtual bool ParseOne(std::string line, LlbExample& example) {
    return false;
  };
};
}  // namespace data

}  // namespace radish
