'''
File: cut.py
Project: code
File Created: Sunday, 17th January 2021 3:53:27 pm
-----------
Last Modified: Sunday, 17th January 2021 3:53:34 pm
Copyright 2020 - 2021 XDU, XDU
-----------
Description: 切分原始数据，原始数据怎么切，mask 数据就怎么切
'''
import numpy as np
import os, json, random
import utils
from PIL import Image, ImageFile
from itertools import groupby
from operator import itemgetter
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sum_json(json_path, json_path_sum):
    '''
    一个图片可能有多处损坏，将损坏信息集中到一张图片中
    '''
    d = {}
    with open(json_path, 'r') as f:
        d = json.load(f)
    # 排序
    d_sort = sorted(d, key=itemgetter('name', 'category', 'bbox'))
    result = []
    # 分组整合
    for name, items in groupby(d_sort, key=itemgetter('name')):
        d_save = defaultdict(list)
        d_save['name'] = name
        for i in items:
            d_save["image_height"] = i['image_height']
            d_save["image_width"] = i['image_width']
            d_save['category'].append(i['category'])
            d_save['bbox'].append(i['bbox'])
        result.append(d_save)
    with open(json_path_sum, 'w') as f:
        json.dump(result, f, indent=4)


def cut_pic(json_path, save_path, cut_num, size, ismask):
    '''
    切分图片，原始图片怎么切，mask 图片就怎么切
    '''
    d = {}
    ls = []
    cnt = 0
    with open(json_path, 'r') as f:
        d = json.load(f)
    
    # 保存文件的路径
    data_path = '../MaskRCNN/maskdata/images/'
    mask_path = '../MaskRCNN/maskdata/masks/'

    # 遍历图片
    for item in d:

        name = item['name']
        # 打开原始图片
        image = Image.open(data_path + name).convert("RGB")
        # 打开 mask 图片
        if ismask:
            mask = Image.open(mask_path + 'Mask_' + name[0:-4] + '.png').convert("RGB")
        height, width = item['image_height'], item['image_width']
        # 遍历一个图片里面所有的缺陷区域
        # 记录这是第几个盒子
        idx = 0
        for cate, box in zip(item['category'], item['bbox']):
            # 坐标
            x0, y0 = box[0], box[1]
            x1, y1 = box[2], box[3]
            # 裁剪 cut_num 张图片 这里做类别平衡处理
            # 每个类 1000 张
            cut_num_ = None
            if cut_num >= 300 and (cate == 1 or cate == 6):
                if cate == 1:
                    cut_num_ = 3
                if cate == 6:
                    cut_num_ = 4
            else:
                cut_num_ = cut_num
            for j in range(cut_num_):
                # 图片命名：图片名_第几张_类别_第几个盒子，防止覆盖文件
                pic_name = name[0:-4] + '_' + str(j) + '_' + str(cate) + '_' + str(idx) + '.png'
                image_path = save_path + 'images/' + pic_name
                if ismask:
                    mask_image_path = save_path + 'masks/' + pic_name
                # 如果存在就不用裁剪，省得浪费时间
                if os.path.exists(image_path):
                    cnt += 1 
                    print(image_path)
                    continue
                # 保存的 json
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
                    a, b = int(x1) + 1 - size, int(x0) - 1
                    # 防止 a 大于等于 b
                    while a >= b:
                        a -= 5
                    left = random.randint(a, b)

                if y1 - y0 <= size and y1 + 1 >= size:
                    a, b = int(y1) + 1 - size, int(y0) - 1
                    while a >= b:
                        a -= 5
                    top = random.randint(a, b)

                # 开始裁剪 正样本
                # 如果横着越出边界
                if left + size > width:
                    left = width - size

                # 竖着越出边界
                if top + size > height:
                    top = height - size

                # 判断裁剪区域和目标区域有无交集，没有交集就目标区域
                if (x1 <= left and y1 <= top) or (x0 >= left + size and y1 <= top) or \
                    (x1 <= left and y0 >= top + size) or (x0 >= left + size and y0 >= top + size):
                    print('------In cut resion, no target------')
                    if x1 <= left and y1 <= top:
                        while x1 <= left and y1 <= top:
                            left -= 5
                            top -= 5
                    elif x0 >= left + size and y1 <= top:
                        while x0 >= left + size and y1 <= top:
                            left += 5
                            top -= 5
                    elif x1 <= left and y0 >= top + size:
                        while x1 <= left and y0 >= top + size:
                            left -= 5
                            top += 5
                    elif x0 >= left + size and y0 >= top + size:
                        while x0 >= left + size and y0 >= top + size:
                            left += 5
                            top += 5

                region = image.crop((left, top, left + size, top + size))
                if ismask:
                    mask_region = mask.crop((left, top, left + size, top + size))

                    # 检验，如果只切割了背景，异常中断
                    a1 = np.array(mask_region)
                    assert len(np.unique(a1)) > 1, mask_image_path
                    mask_region.save(mask_image_path)

                region.save(image_path)
                print(cnt, '/', 2400)
                # 遍历完一张图片，自增
                cnt += 1

                dict_p['name'] = pic_name
                dict_p['category'] = cate
                # 保留残缺区域的相对位置
                dict_p['bbox'] = [
                    round(x0 - left, 2), 
                    round(y0 - top, 2), 
                    round(x1 - left, 2), 
                    round(y1 - top, 2)
                ]
                ls.append(dict_p)
            idx += 1

    with open(save_path + 'cut_data.json', 'w') as f:
        json.dump(ls, f, indent=4)


if __name__ == "__main__":

    # 每个类选择 NUM 张
    NUM = 1000
    # 原始数据的 json
    JSONPATH = 'train_annos.json'
    # json 文件汇总，即一个图片里面有好几个损坏,把他们整合到一起
    JSONPATHSUM = 'cut_{}/cut_{}_sum.json'.format(NUM, NUM)
    # 采样的 json
    SAMPLEPATH = 'cut_{}/cut_{}_sample.json'.format(NUM, NUM)
    # 图片保存路径
    SAVEPATH = 'cut_{}/'.format(NUM)
    # 目标区域裁剪几张
    CUTNUM = 1
    # 目标区域的尺寸
    SIZE = 512

    # 返回要裁剪的 json 的路径
    cut_json_path = utils.select_pic(json_path=JSONPATH, sample_path=SAMPLEPATH, num=NUM, all=False)

    # 汇总
    if not os.path.exists(JSONPATHSUM):
        print('汇总 json 文件')
        sum_json(json_path=cut_json_path, json_path_sum=JSONPATHSUM)

    # 按照汇总好的数据开始切
    cut_pic(json_path=JSONPATHSUM, save_path=SAVEPATH, cut_num=CUTNUM, size=SIZE, ismask=True)
    print('Fucking end.')
# 157001