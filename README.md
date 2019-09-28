# Radish
Radish可以让你的模型从训练到部署都使用相同C++代码库， 借助libtorch, 让你专注实现模型及对应数据处理。


# 为什么造这个轮子

 1)  AI真正的落地需要很好的工程化

 2) 模型太多了，训练， 预处理等也需要很好工程化

 3) 实时训练场景如有些有些RL需要真正多线程支持，而不是Python

 4) 训练与推理相同代码库，缩小落地Gap

如果你碰到以上问题，Radish值得尝试!

# 如何使用

1) 派生自radish::LlbModel类， 实现对应forward过程，以及计算loss的逻辑
2) 决定你的样本特征，以及对应target
3) 实现radish::data::ExampleParser , 根据需要实现对应解析方法
4) 借助radish::train::LlbTrainer 指定对应模板参数，函数参数训练模型
5) ....
   
   可参考bert目录下spanbert以及albert示例。

# 关于ALBERT

论文给出的实验报告，可以看出主要是hidden size在起作用， 共享参数反而使得效果打折扣。
所以本示例实现没有加入参数共享。可自行更改对应代码， 也欢迎pull request.



#  数据载入

你可以使用2种数据格式，一种是基于leveldb, 另一种基于纯文本（一行一个样本)

