# 题目背景

[小目标的异常点检测](https://tianchi.aliyun.com/competition/entrance/531846/information)，超乎我对神经网络实际项目的认知。

# 项目说明

- `code`文件夹下是数据预处理的代码
- `FasterRCNN`文件夹下是模型代码
- `draft`是打草稿的代码

使用时请注意调整项目和数据的路径。

## 数据集

- [初赛训练数据，含标签](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_train_20201231.zip)，e3b9d049f8f15f7a58a6caaae2ae2bf8
- [初赛测试数据，无标签](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_testA_20201231.zip)，b5ac7e51a8fbcb69bb9c36355302d71a

训练集图片，`json` 长度为：15230 ，故有 15230 个需要检测的 `box`，图片数量为：5388。

## 经验总结

### 数据预处理

1. [Python使用PIL库制作mask数据时，JPG与PNG的异同](https://muyuuuu.github.io/2021/01/19/Mask-PIL-jpg-and-png/)
2. [COCO，mask，切割等](https://muyuuuu.github.io/2021/01/21/object-detection-data-preprocess/)