/*
 * File: crf.cc
 * Project: layers
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-07 9:26:09
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#include "layers/crf.h"

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "torch/types.h"
#include "torch/utils.h"

namespace radish {

CRFOptions::CRFOptions(int64_t numTags) : num_tag_(numTags) {}

CRFImpl::CRFImpl(CRFOptions options_) : options(options_) { reset(); }

void CRFImpl::reset() {
  transitions = register_parameter(
      "transitions", torch::empty({options.num_tag_, options.num_tag_}));
  start_transition =
      register_parameter("start_transition", torch::empty({options.num_tag_}));
  end_transition =
      register_parameter("end_transition", torch::empty({options.num_tag_}));
  torch::NoGradGuard guard;
  transitions.normal_(-0.1, 0.1);
  start_transition.uniform_(-0.1, 0.1);
  end_transition.uniform_(-0.1, 0.1);
}

void CRFImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CRF(num_tag=" << options.num_tag_ << ")";
}

Tensor CRFImpl::forward(const Tensor& tags, const Tensor& emissions,
                        const Tensor& theMask) {
  Tensor actualTag = tags.sub(1).transpose_(0, 1);
  Tensor logits = emissions.transpose(0, 1);
  Tensor mask = theMask;
  if (mask.numel() > 0) {
    mask.transpose_(0, 1);
  } else {
    mask = actualTag.ne(0);
  }
  _validate(actualTag, logits, mask);
  //  shape: (batch_size,)
  Tensor numerator = _compute_score(logits, actualTag, mask);
  // shape: (batch_size,)
  Tensor denominator = _compute_normalizer(logits, mask);
  // shape: (batch_size,)
  Tensor llh = numerator - denominator;
  return llh.masked_select(mask).mean();
}

std::vector<std::vector<int>> CRFImpl::decode(const Tensor& emissions,
                                              const Tensor& theMask) {
  Tensor logits = emissions.transpose(0, 1);
  Tensor mask = theMask.transpose(0, 1);
  _validate(mask, logits);
  return _viterbi_decode(logits, mask);
}

void CRFImpl::_validate(const Tensor& mask, const Tensor& emissions,
                        const Tensor& tags) {
  if (tags.numel() > 0) {
    CHECK(tags.dim() == 2);
    CHECK_EQ(tags.size(0), emissions.size(0));
    CHECK_EQ(tags.size(1), emissions.size(1));
  }
  CHECK(emissions.dim() == 3);
  CHECK_EQ(emissions.size(2), options.num_tag_);
  CHECK(mask.select(0, 0).all().item().to<bool>());
}
Tensor CRFImpl::_compute_score(const Tensor& tags, const Tensor& emissions,
                               const Tensor& mask) {
  int64_t seqLen = tags.size(0);
  int64_t batchSz = tags.size(1);
  //
  Tensor score = start_transition.index(tags.select(0, 0));
  score.add_(emissions.select(0, 0).index(tags.select(0, 0)));
  for (auto i = 1; i < seqLen; i++) {
    score.add_(transitions.index({tags.select(0, i - 1), tags.select(0, i)}))
        .mul_(mask.select(0, i));
    auto logits = emissions.select(0, i);
    score.add_(logits.index({torch::arange(batchSz), tags.select(0, i)}))
        .mul_(mask.select(0, i));
  }
  auto seq_ends = mask.sum(0) - 1;
  auto last_tags = tags.index({seq_ends, torch::arange(batchSz)});
  score.add_(end_transition.index({last_tags}));
  return score;
}
Tensor CRFImpl::_compute_normalizer(const Tensor& logits, const Tensor& mask) {
  int64_t seqLen = logits.size(0);
  // (B, Ntag)
  Tensor score = start_transition.add(logits.select(0, 0));
  for (auto i = 1; i < seqLen; i++) {
    // [B,Ntag, 1]
    auto broadcast_score = score.unsqueeze(2);
    //[B,1,Ntag]
    auto broadcast_logits = logits.select(0, i).unsqueeze(1);
    // Compute the score tensor of size (batch_size, num_tags, num_tags) where
    // for each sample, entry at row i and column j stores the sum of scores of
    // all possible tag sequences so far that end with transitioning from tag i
    // to tag j and emitting shape: (B, Ntag, Ntag)
    auto next_score = broadcast_score + transitions + broadcast_logits;
    // Sum over all possible current tags, but we're in score space, so a sum
    // becomes a log-sum-exp: for each sample, entry i stores the sum of scores
    // of all possible tag sequences so far, that end in tag i shape: (B, Ntag)
    next_score = torch::logsumexp(next_score, {1});
    score = torch::where(mask.select(0, i).unsqueeze(1), next_score, score);
  }
  score.add_(end_transition);
  return torch::logsumexp(score, {1});
}

std::vector<std::vector<int>> CRFImpl::_viterbi_decode(const Tensor& logits,
                                                       const Tensor& mask) {
  int64_t seqLen = logits.size(0);
  int64_t batchSz = logits.size(1);
  // (B, Ntag)
  Tensor score = start_transition.add(logits.select(0, 0));
  std::vector<Tensor> history;

  // score is a tensor of size (batch_size, num_tags) where for every batch,
  // value at column j stores the score of the best tag sequence so far that
  // ends with tag j history saves where the best tags candidate transitioned
  // from; this is used when we trace back the best tag sequence

  // Viterbi algorithm recursive case: we compute the score of the best tag
  // sequence for every possible next tag
  for (auto i = 1; i < seqLen; i++) {
    // [B,Ntag, 1]
    auto broadcast_score = score.unsqueeze(2);
    //[B,1,Ntag]
    auto broadcast_logits = logits.select(0, i).unsqueeze(1);
    // Compute the score tensor of size (batch_size, num_tags, num_tags) where
    // for each sample, entry at row i and column j stores the sum of scores of
    // all possible tag sequences so far that end with transitioning from tag i
    // to tag j and emitting shape: (B, Ntag, Ntag)
    auto next_score = broadcast_score + transitions + broadcast_logits;
    auto [max_score, indexies] = next_score.max(1);

    score = torch::where(mask.select(0, i).unsqueeze(1), max_score, score);
    history.push_back(indexies);
  }
  score.add_(end_transition);
  // Now, compute the best path for each sample
  auto seq_ends = mask.sum(0) - 1;
  std::vector<std::vector<int>> rets;
  for (auto i = 0; i < batchSz; i++) {
    std::vector<int> best_tags;
    //  Find the tag which maximizes the score at the last timestep; this is our
    //  best tag for the last timestep
    auto [_, best_last_tag] = score.select(0, i).max(0);
    best_tags.push_back(best_last_tag.item().to<int>());
    // We trace back where the best last tag comes from, append that to our best
    // tag sequence, and trace it back again, and so on
    int seqEnd = seq_ends.select(0, i).item().to<int>();
    for (auto t = seqEnd - 1; t >= 0; t--) {
      auto hist = history[t];
      int lastBestTag = best_tags.back();
      lastBestTag = hist.select(0, i).select(0, lastBestTag).item().to<int>();
      best_tags.push_back(lastBestTag);
    }
    std::reverse(best_tags.begin(), best_tags.end());
    rets.push_back(best_tags);
  }
  return rets;
}

}  // namespace radish
