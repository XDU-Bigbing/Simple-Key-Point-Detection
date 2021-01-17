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
import json, sys
from PIL import Image
from itertools import groupby
from operator import itemgetter


def generate_mask(json_path, save_path, colors):
    '''
    对标注的异常区域生成掩码文件
    '''
    cnt = 1
    with open(json_path, 'r') as f:
        d = json.load(f)
        # 有多个图片，排序
        d_sort = sorted(d, key=itemgetter('name', 'category', 'bbox'))
        for name, items in groupby(d_sort, key=itemgetter('name')):
            height, width = None, None
            # 只能遍历一次
            for pic in items:
                if height is None and width is None:
                    height, width = pic['image_height'], pic['image_width']
                    im = Image.new('RGB', (height, width), colors['background'])
                # 按照 size 创建背景颜色的图片
                x0, y0 = int(pic['bbox'][0]), int(pic['bbox'][1])
                x1, y1 = int(pic['bbox'][2]), int(pic['bbox'][3])
                # 对异常区域创建矩形 作为掩码
                im1 = Image.new('RGB', (x1-x0, y1-y0), colors[str(pic['category'])])
                im.paste(im1, (x0, y0))
                print(str(pic['category']), end=', ')

            im.save(save_path + 'Mask_' + name)
            print("{}%".format(cnt))
            cnt += 1


if __name__ == "__main__":

    # 生成 mask 图片
    JSONPATH = 'maskdata/train_annos.json'
    SAVEPATH = 'maskdata/masks/'
    # 每个类对应不同的颜色
    colors = {
        'background': '#000000',
        '1': '#FF0000',
        '2': '#00FF00',
        '3': '#0000FF',
        '4': '#FFFF00',
        '5': '#FF00FF',
        '6': '#00FFFF',
    }

    generate_mask(json_path=JSONPATH, save_path=SAVEPATH, colors=colors)
