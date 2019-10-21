[![Build Status](https://travis-ci.org/LieluoboAi/radish.png?branch=master)](https://travis-ci.org/LieluoboAi/radish)

# Radish
Radish可以让你的模型从训练到部署都使用相同C++代码库， 借助libtorch, 让你专注实现模型及对应数据处理。

# 如何构建

1) 安装bazel 0.28+
2)  C++17 特性支持的编译器 (7.3.2, 8.3.0已验证)
3) 运行构建比如： bazel build bert:train_albert_main

# 为什么造这个轮子

 1)  AI真正的落地需要很好的工程化

 2) 模型太多了，训练， 预处理等也需要很好工程化

 3) 实时训练场景如有些RL需要真正多线程支持，而不是Python

 4) 训练与推理相同代码库，缩小落地Gap

如果你碰到以上问题，Radish值得尝试!

# 如何使用

1) 派生自radish::LlbModel类， 实现对应forward过程，以及计算loss的逻辑
2) 决定你的样本特征，以及对应target
3) 实现radish::data::ExampleParser , 根据需要实现对应解析方法
4) 借助radish:: train ::LlbTrainer 指定对应模板参数，函数参数训练模型
5) ....
   
   可参考bert目录下spanbert以及albert示例。


#  数据载入

你可以使用2种数据格式，一种是基于leveldb, 另一种基于纯文本（一行一个样本)
基于leveldb的支持完全随机访问， 基于txt的支持多文件输入，每次随机从某文件读入数据



# 使用Goolge BERT  Base Chinese 预训练模型

目前Radish 内置一个兼容BERT实现，可以载入Bert base Chinese模型，模型文件下载
百度网盘：
```
链接:  https://pan.baidu.com/s/1Nlyvw41SfmNzQwhorfQg3g  
提取码:  9m55 
```
radish/bert/finetune/train_bert_cls_finetune.cc 是一个finetune的示例，使用bert base chinese, batch size=32, lr=0.00005 1个epoch,在xnli数据测试集上达到95.94%



# 参考

1)  [Pytorch C++ Doc](https://pytorch.org/cppdocs/ "Pytorch Cpp doc")
2)  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805 "BERT Paper")
3)  [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://openreview.net/forum?id=H1eA7AEtvS "ALBert Paper")
4)  [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529 "SpanBert Paper")




