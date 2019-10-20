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
               const std::string& ignore_name_regex = "",
               torch::Device device = torch::kCPU, bool log = false);

void LoadModelEx(std::shared_ptr<torch::nn::Module> module,
                 const std::string& file_name,
                 const std::string& prefixVarName = "",
                 torch::Device device = torch::kCPU);

/* Correspondig Python export code
 * raw_state_dict = {}
 * for k, v in model.state_dict().items():
 *     if isinstance(v, torch.Tensor):
 *         raw_state_dict[k] = (list(v.size()), v.numpy().tolist())
 *     else:
 *         print("State parameter type error : {}".format(k))
 *         exit(-1)
 *
 * with open('mask_rcnn_coco.json', 'w') as outfile:
 *     json.dump(raw_state_dict, outfile)
 */

torch::OrderedDict<std::string, torch::Tensor> LoadStateDictJson(
    const std::string& file_name);

void LoadStateDictJson(std::shared_ptr<torch::nn::Module> module,
                       const std::string& file_name);

}  // namespace train
}  // namespace  radish
