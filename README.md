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
|-----coco.py                                 制作 coco 数据的 json
|-----utils.py                                小工具
|-----cut_pic.py                              切割图片
|-----view.py                                 标注图片的异常区域，观察
|-----data /                                  训练数据的文件夹
|      |-----train_annos.json                 题目提供的 json
|      |-----annotations /                    制作的 coco json
|            |-----instances_trainval.json   
|      |-----trainval /                       题目的训练数据
|            |-----*.jpg
|-----CUT_PIC /                               切割的图片
|            |-----data.json                  表明切割数据的 json
|            |-----imgs /                     切割的图片
|                    |----- *.jpg
|            |-----masks /                    切割图片异常区域的 mask
|                    |----- *.jpg
|-----view /                                  标注图片的异常区域，观察结果的图片
|      |----- *.jpg
```