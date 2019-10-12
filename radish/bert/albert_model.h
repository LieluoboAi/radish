
#pragma once

#include "radish/layers/layer_norm.h"
#include "radish/train/llb_model.h"
#include "radish/transformer/transformer_encoder.h"
namespace radish {
using Tensor = torch::Tensor;

/// Options for the `ALBert` module.
struct TORCH_API ALBertOptions {
  ALBertOptions(int64_t n_src_vocab);
  TORCH_ARG(int64_t, n_src_vocab);
  TORCH_ARG(int64_t, len_max_seq)=521;
  TORCH_ARG(int64_t, d_word_vec)=128;
  TORCH_ARG(int64_t, n_layers)=5;
  TORCH_ARG(int64_t, n_head)=8;
  TORCH_ARG(int64_t, d_k)=40;
  TORCH_ARG(int64_t, d_v)=40;
  TORCH_ARG(int64_t, d_model)=320;
  TORCH_ARG(int64_t, d_inner)=1280;
  TORCH_ARG(double, dropout) = 0.1;
};

class TORCH_API ALBertModelImpl : public train::LlbModel {
 public:
  ALBertModelImpl(int64_t n_src_vocab)
      : ALBertModelImpl(
            ALBertOptions(n_src_vocab)) {}
  explicit ALBertModelImpl(ALBertOptions options);

  Tensor CalcLoss(const std::vector<Tensor> &examples,
                                      const Tensor &logits,
                                      std::vector<float>& evals,
                                      const Tensor &target = {},
                                      bool train = true) override;

  Tensor forward(std::vector<Tensor> inputs) override;

  ALBertOptions options;
  TransformerEncoder encoder = nullptr;
  LayerNorm laynorm = nullptr;
  torch::nn::Linear vocab_proj = nullptr;
  torch::nn::Linear order_proj = nullptr;
};

TORCH_MODULE(ALBertModel);
}  // namespace radish
