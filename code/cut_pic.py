"""
File: cut_pic.py
Project: code
File Created: Wednesday, 13th January 2021 4:03:59 pm
-----------
Last Modified: Wednesday, 13th January 2021 4:05:03 pm
Copyright 2020 - 2021 XDU, XDU
-----------
Description: 
    - 将切分 图片 和 json 文件保存到 cut_pic 文件夹
    - 将图片切成 512 X 512 均匀大小
    - 正样本含目标区域，负样本不含目标区域
    - 生成 json 文件，用来指明图片是否为正样本、文件名、目标区域、类别
"""

import utils
import random, json
from PIL import Image


def select_pic(sample_num, json_path, data_path, all=False):
    """
    选择要被裁剪的图片
    all == True 时，裁剪所有图片
    """

    if all == False:
        d = utils.select_pic(json_path=json_path, data_path=data_path)
        return d

    if all == True:
        # 打开全部文件，裁剪
        pass


def sample(selected, cut_num, size, json_path, data_path, save_path):
    """
    采样 含异常区域的成为正样本 不含异常区域的称为负样本
    """
    ls = []
    # 对索引编号
    name = 0
    bbox = 1
    image_height = 3
    image_width = 4
    cnt = 0
    # 字典 正确
    for key, value in selected.items():
        for i, _ in enumerate(value):
            with Image.open(data_path + value[i][name]) as im:
                # 获取四个顶点
                x0, y0 = value[i][bbox][0], value[i][bbox][1]
                x1, y1 = value[i][bbox][2], value[i][bbox][3]

                # 产生随机整数，涵盖此区域几个
                for j in range(cut_num):
                    # 保存正样本
                    dict_p = {}
                    # 以 top 和 left 为起点进行裁剪
                    left, top = None, None
                    # 从 [left, top] 开始裁剪
                    # 裁剪大小为 [left->left+size, top->top+size]
                    
                    # 如果损坏区域大于 size
                    if y1 - y0 > size:
                        top = random.randint(int(y0) + 10 - size, int(y0))
                    if x1 - x0 > size:
                        left = random.randint(int(x0) + 10 - size, int(x0))

                    # 如果顶点左侧越出边界
                    if x1 + 1 - size < 0:
                        left = random.randint(0, int(x0))

                    # 如果顶点下侧越出边界
                    if y1 + 1 - size < 0:
                        top = random.randint(0, int(y0))
                    
                    # 不满足以上任何情况
                    if x1 - x0 <= size and x1 + 1 >= size:
                        left = random.randint(int(x1) + 1 - size, int(x0) - 1)

                    if y1 - y0 <= size and y1 + 1 >= size:
                        top = random.randint(int(y1) + 1 - size, int(y0) - 1)

                    # 开始裁剪 正样本
                    # 如果横着越出边界
                    if left + size > value[i][image_width]:
                        left = value[i][image_width] - size

                    # 竖着越出边界
                    if top + size > value[i][image_height]:
                        top = value[i][image_height] - size
                    
                    print(x0, left, y0, top, x0-left, y0-top, sep=', ')

                    region = im.crop((left, top, left + size, top + size))
                    # P 表示正样本
                    region.save(save_path + 'imgs/' + 'P_' + str(j) + '_' + value[i][name])
                    dict_p['name'] = 'P_' + str(j) + '_' + value[i][name]
                    dict_p['positive'] = True
                    dict_p['category'] = key
                    # 保留残缺区域的相对位置
                    dict_p['bbox'] = [
                        round(x0 - left, 2), 
                        round(y0 - top, 2), 
                        round(x1 - left, 2), 
                        round(y1 - top, 2)
                    ]
                    ls.append(dict_p)
                    # 保存负样本
                    # dict_n = {}

    return ls


def generate_mask(json_path, data_path, save_path, colors, size):
    '''
    对标注的异常区域生成掩码文件
    '''
    with open(json_path, 'r') as f:
        d = json.load(f)
        # 遍历图片
        for i in range(len(d)):
            # 按照 size 创建背景颜色的图片
            im = Image.new('RGB', (size, size), colors['background'])
            x0, y0 = int(d[i]['bbox'][0]), int(d[i]['bbox'][1])
            x1, y1 = int(d[i]['bbox'][2]), int(d[i]['bbox'][3])
            # 对异常区域创建矩形 作为掩码
            im1 = Image.new('RGB', (x1-x0, y1-y0), colors[d[i]['category']])
            im.paste(im1, (x0, y0))
            im.save(save_path + 'masks/Mask_' + d[i]['name'])


if __name__ == "__main__":

    JSONPATH = 'data/train_annos.json'
    DATAPATH = "data/trainval/"
    SAVEPATH = 'CUT_PIC/'

    # 每个类选择 10 张图片
    sample_num = 10
    # 选择要裁剪的图片
    selected = select_pic(sample_num=sample_num, json_path=JSONPATH,
                          data_path=DATAPATH, all=False)

    # 正负样本各裁剪几张
    cut_num = 3
    # 裁剪的尺寸 [size X size]
    size = 512
    # 裁剪得到的样本
    json_dict = sample(selected, cut_num=cut_num, size=size,
                     json_path=JSONPATH, data_path=DATAPATH, save_path=SAVEPATH)

    # 保存 json
    with open(SAVEPATH + 'data.json', 'w') as f:
        json.dump(json_dict, f, indent=4)

    # 生成 mask 图片
    JSONPATH = SAVEPATH + 'data.json'
    DATAPATH = SAVEPATH + 'imgs/'
    # 每个类对应不同的颜色
    colors = {
        'background': '#FFFFFF',
        '1': '#FF0000',
        '2': '#00FF00',
        '3': '#0000FF',
        '4': '#FFFF00',
        '5': '#FF00FF',
        '6': '#00FFFF',
    }

    generate_mask(json_path=JSONPATH, data_path=DATAPATH, 
                  save_path=SAVEPATH, colors=colors, size=size)
