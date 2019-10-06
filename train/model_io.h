/*
 * File: model_io.h
 * Project: train
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-24 2:53:28
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include <iostream>
#include <regex>

#include "torch/torch.h"

namespace radish {
namespace train {
void SaveModel(std::shared_ptr<torch::nn::Module> module,
               const std::string& file_name);

void LoadModel(std::shared_ptr<torch::nn::Module> module,
               const std::string& file_name,
               const std::string& ignore_name_regex = "");

void LoadModelEx(std::shared_ptr<torch::nn::Module> module,
               const std::string& file_name,
               const std::string& prefixVarName = "");
}  // namespace train
}  // namespace  radish
