/*
 * File: bert_options.cc
 * Project: model
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-18 11:07:47
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "radish/bert/model/bert_options.h"

namespace radish {

BertOptions BertOptions::kBertBaseOpts = BertOptions(21128, 768)
                                             .intermediate_size(3072)
                                             .max_pos(512)
                                             .max_types(2)
                                             .num_heads(12)
                                             .num_layers(12)
                                             .dropout(0.1);
BertOptions BertOptions::kMiniAlbertOpts = BertOptions(21128, 240)
                                             .intermediate_size(960)
                                             .max_pos(512)
                                             .max_types(2)
                                             .num_heads(8)
                                             .num_layers(5)
                                             .need_factor_embedding(true)
                                             .d_wordvec(128)
                                             .repeat_stochastic_layers(2)
                                             .dropout(0.1);
}  // namespace radish
