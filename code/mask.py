'''
File: mask.py
Project: code
File Created: Sunday, 17th January 2021 2:53:57 pm
-----------
Last Modified: Sunday, 17th January 2021 2:54:10 pm
Copyright 2020 - 2021 XDU, XDU
-----------
Description: https://github.com/XDU-Bigbing/Simple-Key-Point-Detection/issues/1#issuecomment-761743786
'''
import json, os
import numpy as np
from PIL import Image
from itertools import groupby
from operator import itemgetter


def generate_mask(json_path, save_path, colors):
    '''
    对标注的异常区域生成掩码文件
    '''
    cnt = 1
    d = {}
    with open(json_path, 'r') as f:
        d = json.load(f)
        # 有多个图片，排序
    d_sort = sorted(d, key=itemgetter('name', 'category', 'bbox'))
    for name, items in groupby(d_sort, key=itemgetter('name')):
        if os.path.exists(save_path + 'Mask_' + name[:-4] + '.png'):
            cnt += 1
            continue
        height, width = None, None
        # 只能遍历一次
        # 变量作用域
        im = Image.new('RGB', (10, 10), colors['background'])
        for pic in items:
            if height is None and width is None:
                height, width = pic['image_height'], pic['image_width']
                im = Image.new('RGB', (width, height), colors['background'])
            # 按照 size 创建背景颜色的图片
            x0, y0 = int(pic['bbox'][0]), int(pic['bbox'][1])
            x1, y1 = int(pic['bbox'][2]), int(pic['bbox'][3])
            # 对异常区域创建矩形 作为掩码
            im1 = Image.new('RGB', (x1-x0, y1-y0), colors[str(pic['category'])])
            im.paste(im1, (x0, y0))
        # 太慢了
        # num = np.unique(np.array(im))
        # assert num.size > 1, 'mask only have background'
        # png 保留原始颜色
        im.save(save_path + 'Mask_' + name[:-4] + '.png')
        # 抽查
        if cnt % 17 == 0 and cnt != 0:
            a = np.array(im)
            assert len(np.unique(a)) > 0
        print(cnt)
        cnt += 1
        if cnt >= 500:
            break


if __name__ == "__main__":

    # 生成 mask 图片
    JSONPATH = 'maskdata/train_annos.json'
    SAVEPATH = 'maskdata/masks/'
    # 每个类对应不同的颜色
    colors = {
        'background': '#000000',
        '1': '#110000',
        '2': '#002200',
        '3': '#000033',
        '4': '#445500',
        '5': '#660077',
        '6': '#008899',
    }

    generate_mask(json_path=JSONPATH, save_path=SAVEPATH, colors=colors)
    print('Fucking End!!!')