/*
 * File: bert_tokenizer_test.cc
 * Project: bert
 * File Created: Saturday, 19th October 2019 6:17:51 pm
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Saturday, 19th October 2019 6:20:55 pm
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */
#include <memory>

#include "gtest/gtest.h"

#include "radish/utils/text_tokenizer.h"

class BertTokenizerTest : public testing::Test {
 protected:
  virtual void SetUp() {
    tokenizer_.reset(
        radish::TextTokenizerFactory::Create("radish::BertTokenizer"));
    EXPECT_TRUE(tokenizer_->Init("/e/data/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"));
  }
  virtual void TearDown() {}
  std::unique_ptr<radish::TextTokenizer> tokenizer_;
};

TEST_F(BertTokenizerTest, TestEncode) {
  auto ids = tokenizer_->Encode(
      "欢迎语音识别开源工具Kaldi的创始人，语音和AI领域大佬 Daniel Povey "
      "加入小米！");
  std::vector<int> expected = {
      3614, 6816, 6427, 7509, 6399, 1166, 2458, 3975, 2339, 1072, 11306, 8635,
      8169, 4638, 1158, 1993, 782,  8024, 6427, 7509, 1469, 8578, 7566,  1818,
      1920, 878,  9701, 8559, 8519, 8179, 1217, 1057, 2207, 5101, 8013};
  EXPECT_EQ(ids, expected);
  ids = tokenizer_->Encode(
      "4.结账输入专属优惠券：UAADUOMAI10819D11   还可以再次享受满300减30。");
  expected = {125,  119,  5310, 6572,  6783, 1057, 683,  2247, 831,
              2669, 1171, 8038, 11107, 8695, 9731, 8404, 8169, 8311,
              9313, 8160, 8168, 8452,  6820, 1377, 809,  1086, 3613,
              775,  1358, 4007, 8209,  1121, 8114, 511};
  EXPECT_EQ(ids, expected);
  ids = tokenizer_->Encode("`%￥#@×&+——？、‘【】");
  expected = {100, 110, 8101, 108, 137, 190, 111, 116,
              100, 100, 8043, 510, 100, 523, 524};
  EXPECT_EQ(ids, expected);
  ids = tokenizer_->Encode("òóôõöø");
  expected = {11383, 8902, 8167, 13364};
  EXPECT_EQ(ids, expected);
}
