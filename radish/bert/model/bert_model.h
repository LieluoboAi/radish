/*
 * File: bert_model.h
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 5:17:56
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once

#include <cstddef>
#include <vector>

#include "radish/bert/model/bert_embedding.h"
#include "radish/bert/model/bert_encoder.h"
#include "radish/bert/model/bert_options.h"

#include "torch/nn/cloneable.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {

class TORCH_API BertModelImpl : public torch::nn::Cloneable<BertModelImpl> {
 public:
  explicit BertModelImpl(const BertOptions& options_);

  void reset() override;

  std::vector<Tensor> forward(Tensor input_ids, Tensor attention_mask = {},
                              Tensor token_type_ids = {},
                              Tensor position_ids = {}, Tensor head_mask = {});

  BertOptions options;
  BertEmbedding embeddings = nullptr;
  BertEncoder encoder = nullptr;
  BertPooler pooler = nullptr;
};

TORCH_MODULE(BertModel);

}  // namespace radish
