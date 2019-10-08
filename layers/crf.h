/*
 * File: crf.h
 * Project: layers
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-10-07 9:12:13
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */

#pragma once
#include <cstddef>
#include <vector>

#include "torch/nn/cloneable.h"
#include "torch/nn/pimpl.h"
#include "torch/types.h"

namespace radish {
using Tensor = ::torch::Tensor;
/// Options for the `Embedding` module.
struct TORCH_API CRFOptions {
  CRFOptions(int64_t numTags);
  /// The number of  different tags
  TORCH_ARG(int64_t, num_tag);
};

class TORCH_API CRFImpl : public torch::nn::Cloneable<CRFImpl> {
 public:
  explicit CRFImpl(int64_t numTag) : CRFImpl(CRFOptions(numTag)) {}
  explicit CRFImpl(CRFOptions options);

  void reset() override;

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  // compute the conditional log likelihood  of  sequence of tags give the
  // emission scores
  //  tensor assumed batch first, tags and padded with zero, so valid tag  index
  //  should start with 1
  Tensor forward(const Tensor& tags, const Tensor& emissions, const Tensor& theMask={});

  // viterbi decode
  std::vector<std::vector<int>>  decode(const Tensor& emissions,const Tensor& mask);

  /// The `Options` used to configure this `CRF` module.
  CRFOptions options;

  Tensor transitions;
  Tensor start_transition;
  Tensor end_transition;
  private:
    void _validate(const Tensor& mask, const Tensor& emissions, const Tensor& tags={});
    Tensor _compute_score(const Tensor& tags, const Tensor& emissions, const Tensor& mask);
    Tensor  _compute_normalizer(const Tensor& logits, const Tensor& mask);
    std::vector<std::vector<int>>  _viterbi_decode(const Tensor& logits, const Tensor& mask);
};

/// A `ModuleHolder` subclass for `CRFImpl`.
/// See the documentation for `CRFImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(CRF);

}  // namespace radish
