/*
 * File: bert_model.cc
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-17 5:25:09
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "radish/bert/model/bert_model.h"

namespace radish {

BertModelImpl::BertModelImpl(const BertOptions& options_) : options(options_) {
  reset();
}

void BertModelImpl::reset() {
  embeddings = BertEmbedding(options);
  register_module("embeddings", embeddings);
  encoder = BertEncoder(options);
  register_module("encoder", encoder);
  pooler = BertPooler(options);
  register_module("pooler", pooler);
}

std::vector<Tensor> BertModelImpl::forward(Tensor input_ids,
                                           Tensor attention_mask,
                                           Tensor token_type_ids,
                                           Tensor position_ids,
                                           Tensor head_mask) {
  if (attention_mask.numel() == 0) {
    attention_mask = torch::ones_like(input_ids);
  }
  if (token_type_ids.numel() == 0) {
    token_type_ids = torch::zeros_like(input_ids);
  }
  // We create a 3D attention mask from a 2D tensor mask.
  // Sizes are [batch_size, 1, 1, to_seq_length]
  // So we can broadcast to [batch_size, num_heads, from_seq_length,
  // to_seq_length] this attention mask is more simple than the triangular
  // masking of causal attention used in OpenAI GPT, we just need to prepare the
  // broadcast dimension here.
  auto extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2);

  // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
  // masked positions, this operation will create a tensor which is 0.0 for
  // positions we want to attend and -10000.0 for masked positions.
  //  Since we are adding it to the raw scores before the softmax, this is
  // effectively the same as removing these entirely.

  extended_attention_mask.neg_().add_(1.0).mul_(-10000.0);

  // Prepare head mask if needed
  // 1.0 in head_mask indicate we keep the head
  // attention_probs has shape bsz x n_heads x N x N
  // input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
  // and head_mask is converted to shape [num_hidden_layers x batch x num_heads
  // x seq_length x seq_length]
  if (head_mask.numel() > 0) {
    if (head_mask.dim() == 1) {
      head_mask =
          head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
      head_mask = head_mask.expand({options.num_layers(), -1, -1, -1, -1});
    } else if (head_mask.dim() == 2) {
      head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
          -1);  // We can specify head_mask for each layer
    }
  }
  auto embedding_output = embeddings(input_ids, position_ids, token_type_ids);
  auto encoder_outputs =
      encoder(embedding_output, extended_attention_mask, head_mask);
  auto sequence_output = encoder_outputs[0];
  auto pooled_output = pooler(sequence_output);
  return {sequence_output, pooled_output};
}

}  // namespace radish
