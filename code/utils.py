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


def select_pic(json_path, sample_path, num, all):
    # 随机选 1000 张图片 
    # 从 1000 张图片里选择 6 个类的 10 张图片
    # 如果全部选择，那么返回原始的 json 路径
    if all:
        return json_path
    else:
        classes = defaultdict(list)
        with open (json_path, 'r') as f:
            dict_ = json.load(f)
            for i in dict_:
                classes[str(i['category'])].append(i)

        for _, value in classes.items():
            print(len(value))
            random.shuffle(value)

        ls = []

        for cate in range(1, 7):
            # print(classes[str(cate)])
            length = None
            if len(classes[str(cate)]) >= num:
                length = num
            else:
                length = len(classes[str(cate)])

            for i in range(length):
            # 添加图片的 名称、位置、分类
                d = {}
                d['name'] = classes[str(cate)][i]['name']
                d['bbox'] = classes[str(cate)][i]['bbox']
                d['category'] =  classes[str(cate)][i]['category']
                d['image_height'] = classes[str(cate)][i]['image_height']
                d['image_width'] = classes[str(cate)][i]['image_width']
                ls.append(d)

        random.shuffle(ls)

    print('Data Number is ', len(ls))

    with open(sample_path, 'w') as f:
        json.dump(ls, f, indent=4)

    return sample_path

def get_image_id(json_path):
    '''
    # 生成每一个图片的唯一 id 
    # 图片名相同，id 相同
    '''
    cnt = 1
    # 图片 id 的保存
    image_id = {}
    # train_data 和 test_data 的 path 的路径
    with open(json_path, 'r') as f:
        d = json.load(f)
        for i in range(len(d)):
            if d[i]['name'] not in image_id.keys():
                image_id[d[i]['name']] = cnt
                cnt += 1

    print('image id number is {}'.format(len(image_id.keys())))
    return image_id
