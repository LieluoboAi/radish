/*
 * File: bert_attention.cc
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 2:09:43
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include "radish/bert/model/bert_attention.h"

#include "torch/nn/init.h"

namespace radish {

BertSelfAttentionImpl::BertSelfAttentionImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}

void BertSelfAttentionImpl::reset() {
  attention_head_size_ = options.hidden_size() / options.num_heads();
  all_head_size_ = options.num_heads() * attention_head_size_;
  query = torch::nn::Linear(
      torch::nn::LinearOptions(options.hidden_size(), all_head_size_));
  register_module("query", query);
  key = torch::nn::Linear(
      torch::nn::LinearOptions(options.hidden_size(), all_head_size_));
  register_module("key", key);
  value = torch::nn::Linear(
      torch::nn::LinearOptions(options.hidden_size(), all_head_size_));
  register_module("value", value);
  dropout = torch::nn::Dropout(options.dropout());
  register_module("dropout", dropout);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(query->weight, 0, options.init_range());
  torch::nn::init::normal_(key->weight, 0, options.init_range());
  torch::nn::init::normal_(value->weight, 0, options.init_range());
  torch::nn::init::constant_(query->bias, 0);
  torch::nn::init::constant_(key->bias, 0);
  torch::nn::init::constant_(value->bias, 0);
}

/// Pretty prints the `BertSelfAttention` module into the given `stream`.
void BertSelfAttentionImpl::pretty_print(std::ostream& stream) const {}

Tensor BertSelfAttentionImpl::transpose_for_scores(Tensor x) {
  int bsz = x.size(0);
  int seqlen = x.size(1);
  torch::IntArrayRef new_x_shape = {bsz, seqlen, options.num_heads(),
                                    attention_head_size_};
  x = x.view(new_x_shape);
  return x.permute({0, 2, 1, 3});
}

std::vector<Tensor> BertSelfAttentionImpl::forward(Tensor hidden_states,
                                                   Tensor attention_mask,
                                                   Tensor head_mask) {
  auto mixed_query_layer = query(hidden_states);
  auto mixed_key_layer = key(hidden_states);
  auto mixed_value_layer = value(hidden_states);

  auto query_layer = transpose_for_scores(mixed_query_layer);
  auto key_layer = transpose_for_scores(mixed_key_layer);
  auto value_layer = transpose_for_scores(mixed_value_layer);

  //  Take the dot product between "query" and "key" to get the raw attention
  //  scores.
  auto attention_scores =
      torch::matmul(query_layer, key_layer.transpose(-1, -2));
  attention_scores = attention_scores / std::sqrt(attention_head_size_);
  if (attention_mask.numel() > 0) {
    // Apply the attention mask is (precomputed for all layers in BertModel
    // forward() function)
    attention_scores.add_(attention_mask);
  }

  // Normalize the attention scores to probabilities.
  auto attention_probs = torch::softmax(attention_scores, 1);

  // This is actually dropping out entire tokens to attend to, which might
  // seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs);

  // Mask heads if we want to
  if (head_mask.numel() > 0) {
    attention_probs.mul_(head_mask);
  }
  auto context_layer = torch::matmul(attention_probs, value_layer);
  context_layer = context_layer.permute({0, 2, 1, 3}).contiguous();

  // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
  context_layer = context_layer.view(
      {context_layer.size(0), context_layer.size(1), all_head_size_});
  if (options.output_attentions()) {
    return {context_layer, attention_probs};
  } else {
    return {context_layer};
  }
}

BertSelfOutputImpl::BertSelfOutputImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}

void BertSelfOutputImpl::reset() {
  dense = torch::nn::Linear(
      torch::nn::LinearOptions(options.hidden_size(), options.hidden_size()));
  register_module("dense", dense);
  layer_norm = LayerNorm(options.hidden_size(), options.ln_eps());
  register_module("LayerNorm", layer_norm);
  dropout = torch::nn::Dropout(options.dropout());
  register_module("dropout", dropout);
  torch::NoGradGuard guard;
  torch::nn::init::normal_(dense->weight, 0, options.init_range());
  torch::nn::init::constant_(dense->bias, 0);
}

Tensor BertSelfOutputImpl::forward(Tensor hidden_states, Tensor input_tensor) {
  hidden_states = dense(hidden_states);
  hidden_states = dropout(hidden_states);
  hidden_states = layer_norm(hidden_states.add(input_tensor));
  return hidden_states;
}

BertAttentionImpl::BertAttentionImpl(const BertOptions& options_)
    : options(options_) {
  reset();
}

void BertAttentionImpl::reset() {
  self = BertSelfAttention(options);
  register_module("self", self);
  output = BertSelfOutput(options);
  register_module("output", output);
}

std::vector<Tensor> BertAttentionImpl::forward(Tensor input_tensor,
                                               Tensor attention_mask,
                                               Tensor head_mask) {
  auto self_outputs = self(input_tensor, attention_mask, head_mask);
  auto attention_output = output(self_outputs[0], input_tensor);
  if (options.output_attentions()) {
    return {attention_output, self_outputs[1]};
  } else {
    return {attention_output};
  }
}
}  // namespace radish
