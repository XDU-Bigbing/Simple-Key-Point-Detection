# 题目背景

[https://tianchi.aliyun.com/competition/entrance/531846/information](小目标的异常点检测)

## 数据集

- [初赛训练数据，含标签](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_train_20201231.zip)，e3b9d049f8f15f7a58a6caaae2ae2bf8
- [初赛测试数据，无标签](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531846/tile_round1_testA_20201231.zip)，b5ac7e51a8fbcb69bb9c36355302d71a

# 运行

- `conda activate GPU-info`
- `cd 1307/TianChi/code`
- `python coco.py`
- `python train.py`

# 项目结构

```
Tianchi
|-----coco.py
|-----utils.py
|-----data/
|      |-----train_annos.json
|      |-----annotations/
|            |-----instances_trainval.json
|      |-----trainval
|            |-----*.jpg

```