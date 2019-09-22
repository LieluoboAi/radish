/*
 * File: test_radam.cc
 * Project: optimization
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-14 11:46:25
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#include <chrono>  //  for high_resolution_clock
#include <iostream>

#include "torch/torch.h"

#include "optimization/radam.h"

using namespace torch;  // NOLINT
using namespace std;    // NOLINT

struct AlexNetImpl : nn::Module {
  explicit AlexNetImpl(int64_t N)
      : conv1(register_module(
            "conv1",
            nn::Conv2d(nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)))),
        conv2(register_module(
            "conv2", nn::Conv2d(nn::Conv2dOptions(64, 192, 5).padding(2)))),
        conv3(register_module(
            "conv3", nn::Conv2d(nn::Conv2dOptions(192, 384, 3).padding(1)))),
        conv4(register_module(
            "conv4", nn::Conv2d(nn::Conv2dOptions(384, 256, 3).padding(1)))),
        conv5(register_module(
            "conv5", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).padding(1)))),
        linear1(register_module("linear1", nn::Linear(9216, 4096))),
        linear2(register_module("linear2", nn::Linear(4096, 4096))),
        linear3(register_module("linear3", nn::Linear(4096, 1000))),
        dropout(
            register_module("dropout", nn::Dropout(nn::DropoutOptions(0.5)))) {}

  torch::Tensor forward(const torch::Tensor& input) {
    auto x = torch::relu(conv1(input));
    x = torch::max_pool2d(x, 3, 2);

    x = relu(conv2(x));
    x = max_pool2d(x, 3, 2);

    x = relu(conv3(x));
    x = relu(conv4(x));
    x = relu(conv5(x));
    x = max_pool2d(x, 3, 2);
    // Classifier, 256 * 6 * 6 = 9216
    x = x.view({x.size(0), 9216});
    x = dropout(x);
    x = relu(linear1(x));

    x = dropout(x);
    x = relu(linear2(x));

    x = linear3(x);
    return x;
  }
  torch::nn::Linear linear1, linear2, linear3;
  nn::Dropout dropout;
  nn::Conv2d conv1, conv2, conv3, conv4, conv5;
};

TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);

int main() {
  torch::Device device = torch::kCPU;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count()
            << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

  int batch_size = 128;
  int iterations = 50;
  auto model = AlexNet(224);
  radish::optim::RAdam optim(model->parameters(),
                             radish::optim::RAdamOptions(1e-3));

  model->train();
  model->to(device);

  torch::Tensor x, target, y, loss;
  target = torch::randn({batch_size, 1000}, device);
  x = torch::ones({batch_size, 3, 224, 224}, device);
  for (int i = 0; i < iterations; ++i) {
    optim.zero_grad();
    y = model->forward(x);
    loss = torch::mse_loss(y, target);
    loss.backward();
    optim.step();
    if (i % 10 == 0) cout << loss << endl;
  }
}
