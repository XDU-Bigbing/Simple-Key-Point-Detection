'''
File: utils.py
Project: code
File Created: Wednesday, 13th January 2021 9:49:12 am
-----------
Last Modified: Wednesday, 13th January 2021 4:10:08 pm
Copyright 2020 - 2021 XDU, XDU-BigBing
-----------
Description: 一些用到的小工具
'''

import json
import random
from collections import defaultdict


def get_data_num(filename):
    '''
    获取一共有多少数据
    '''
    num = 0
    with open(filename, 'r') as f:
        num = len(json.load(f))
    return num


def get_train_test_valid_num(**kwargs):
    '''
    计算训练集、测试集合的数量
    '''
    # int 防止出现浮点数
    data_num = kwargs['data_num']
    train_num = int(data_num * kwargs['train_scale'])
    test_num = int(data_num * kwargs['test_scale'])
    valid_num = int(data_num * kwargs['valid_scale'])
    return train_num, test_num, valid_num


def select_pic(json_path, data_path):
    # 随机选 1000 张图片 
    # 从 1000 张图片里选择 6 个类的 10 张图片
    random.seed(10)
    data = set()
    for i in range(1000):
        data.add(random.randint(0, 15229))
    # 集合字典 键为类别 值为图片
    d = defaultdict(list)
    # 开始选择
    with open (json_path, 'r') as f:
        dict_ = json.load(f)
        for i in data:
            if len(d[str(dict_[i]['category'])]) < 10:
                # 添加图片的 名称、位置、分类
                d[str(dict_[i]['category'])].append([dict_[i]['name'],
                                                    dict_[i]['bbox'],
                                                    dict_[i]['category'],
                                                    dict_[i]['image_height'],
                                                    dict_[i]['image_width']])
    return d
    # 返回字典的形式是：
    # '2': [['220_17_t20201124131932638_CAM2.jpg', [1692.18, 5728.83, 1736.16, 5789.79], 2],
    # ['254_41_t2020113015115013_CAM1.jpg', [6477.93, 5481.99, 6547.93, 5496.99], 2],
    # ['227_104_t20201126091444537_CAM2.jpg', [1826.08, 347.37, 1861.05, 354.45], 2],
    # ['220_54_t20201124133049493_CAM2.jpg', [6739.77, 5307.27, 6820.77, 5462.27], 2],
    # ['230_141_t20201126145759141_CAM3.jpg', [655.36, 3086.72, 676.36, 3103.72], 2],
    # ['223_18_t20201125083337636_CAM3.jpg', [3202.48, 3045.74, 3279.48, 3249.41], 2],
    # ['254_86_t20201130152532909_CAM2.jpg', [6623.08, 5359.19, 6637.68, 5398.95], 2],
    # ['241_3_t2020112812290827_CAM1.jpg', [1511.58, 5441.82, 1530.58, 5456.82], 2],
    # ['223_29_t2020112508364732_CAM2.jpg', [1712.75, 5507.41, 1735.75, 5530.41], 2],
    # ['245_158_t20201128145400712_CAM3.jpg', [3283.95, 716.41, 3408.95, 775.41], 2]],