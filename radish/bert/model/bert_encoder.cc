/*
 * File: bert_encoder.cc
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 4:04:20
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "radish/bert/model/bert_encoder.h"

#include "radish/bert/model/bert_layer.h"
namespace radish {

BertEncoderImpl::BertEncoderImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}
void BertEncoderImpl::reset() {
  layer = torch::nn::ModuleList();
  for (auto i = 0; i < options.num_layers(); i++) {
    layer->push_back(radish::BertLayer(options));
  }
  register_module("layer", layer);
}
std::vector<Tensor> BertEncoderImpl::forward(Tensor hidden_states,
                                             Tensor attention_mask,
                                             Tensor head_mask) {
  std::vector<Tensor> enc_slf_attn_list;
  for (auto i = 0; i < options.num_layers(); i++) {
    auto elayer = layer->ptr<BertLayerImpl>(i);
    std::vector<Tensor> rets =
        elayer->forward(hidden_states, attention_mask, head_mask);
    hidden_states = rets[0];
    const auto& enc_slf_attn = rets[1];
    if (options.output_attentions()) {
      enc_slf_attn_list.push_back(enc_slf_attn);
    }
  }
  if (options.output_attentions()) {
    enc_slf_attn_list.insert(enc_slf_attn_list.begin(), hidden_states);
    return enc_slf_attn_list;
  }
  return {hidden_states};
}

BertPoolerImpl::BertPoolerImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}
void BertPoolerImpl::reset() {
  dense = torch::nn::Linear(options.hidden_size(), options.hidden_size());
  register_module("dense", dense);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(dense->weight, 0, options.init_range());
  torch::nn::init::constant_(dense->bias, 0);
}
Tensor BertPoolerImpl::forward(Tensor hidden_states) {
  auto pool_output = hidden_states.select(1, 0);
  pool_output = dense(pool_output);
  return torch::tanh(pool_output);
}

}  // namespace radish