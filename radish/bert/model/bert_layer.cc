/*
 * File: bert_layer.cc
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 2:44:43
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "radish/bert/model/bert_layer.h"

#include <cmath>
#include "torch/nn/init.h"

namespace radish {

// static Tensor gelu_new(Tensor x) {
//   // Implementation of the gelu activation function currently in Google Bert
//   // repo (identical to OpenAI GPT).
//   //    Also see https://arxiv.org/abs/1606.08415

//   //  0.5 * x * (1 + torch::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 *
//   //  torch::pow(x, 3))));
//   return x * 0.5 * (1.0 + torch::erf(x.div(std::sqrt(2.0))));
// }
BertIntermediateImpl::BertIntermediateImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}
void BertIntermediateImpl::reset() {
  dense = torch::nn::Linear(options.hidden_size(), options.intermediate_size());
  register_module("dense", dense);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(dense->weight, 0, options.init_range());
  torch::nn::init::constant_(dense->bias, 0);
}
Tensor BertIntermediateImpl::forward(Tensor hidden_states) {
  hidden_states = dense(hidden_states);
  hidden_states = torch::gelu(hidden_states);
  return hidden_states;
}

BertOutputImpl::BertOutputImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}

void BertOutputImpl::reset() {
  dense = torch::nn::Linear(options.intermediate_size(), options.hidden_size());
  register_module("dense", dense);
  layer_norm = LayerNorm(options.hidden_size(), options.ln_eps());
  register_module("LayerNorm", layer_norm);
  dropout = torch::nn::Dropout(options.dropout());
  register_module("dropout", dropout);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(dense->weight, 0, options.init_range());
  torch::nn::init::constant_(dense->bias, 0);
}

Tensor BertOutputImpl::forward(Tensor hidden_states, Tensor input_tensor) {
  hidden_states = dense(hidden_states);
  hidden_states = dropout(hidden_states);
  hidden_states.add_(input_tensor);
  hidden_states = layer_norm(hidden_states);
  return hidden_states;
}

BertLayerImpl::BertLayerImpl(const BertOptions& options_) : options(options_) {
  reset();
}

void BertLayerImpl::reset() {
  attention = BertAttention(options);
  register_module("attention", attention);
  intermediate = BertIntermediate(options);
  register_module("intermediate", intermediate);
  output = BertOutput(options);
  register_module("output", output);
}

std::vector<Tensor> BertLayerImpl::forward(Tensor hidden_states,
                                           Tensor attention_mask,
                                           Tensor head_mask) {
  auto attention_outputs = attention(hidden_states, attention_mask, head_mask);
  auto attention_output = attention_outputs[0];
  auto intermediate_output = intermediate(attention_output);
  auto layer_output = output(intermediate_output, attention_output);
  if (options.output_attentions()) {
    return {layer_output, attention_outputs[1]};
  } else {
    return {layer_output};
  }
}

}  // namespace radish
