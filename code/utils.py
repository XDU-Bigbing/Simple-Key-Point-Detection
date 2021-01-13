import json

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
