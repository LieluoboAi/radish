/*
 * File: span_bert_example.cc
 * Project: bert
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-20 10:52:21
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "bert/span_bert_example.h"

namespace knlp {

std::vector<Tensor> SpanBertExample::ToTensorList() { return {}; }
bool SpanBertExample::FromMessage(const train::TrainExample& ex) {
  return true;
}

}  // namespace knlp
