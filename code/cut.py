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


def cut_pic(json_path, save_path, num, scale, cut_num, ismask, adaptive):
    '''
    切分图片，原始图片怎么切，mask 图片就怎么切
    '''
    d = {}
    ls = []
    cnt = 0
    with open(json_path, 'r') as f:
        d = json.load(f)
    
    # 读取文件的路径
    data_path = '../DATA/tile_round1_train_20201231/train_imgs/'
    mask_path = '../DATA/MASK_Train/masks/'

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
            # 裁剪图片的大小
            random_width, random_height = None, None
            # 如果自适应
            if adaptive:
                # 计算裁剪的高度和宽度
                window = random.randint(-3, 4)
                if window == 0:
                    window = 1
                if window < 0:
                    random_width = random.randint(int(x1 - x0) * (scale + window), int(x1 - x0) * scale)
                    random_height = random.randint(int(y1 - y0) * (scale + window), int(y1 - y0) * scale)
                else:
                    random_width = random.randint(int(x1 - x0) * scale, int(x1 - x0) * (scale + window))
                    random_height = random.randint(int(y1 - y0) * scale, int(y1 - y0) * (scale + window))

                # 如果缩放后超出原来图片的大小
                if random_height > height:
                    random_height = height
                if random_width > width:
                    random_width = width
            # 不自适应
            else: 
                random_width, random_height = width, height
            
            # 裁剪 cut_num 张图片 这里做类别平衡处理
            # {'4': 1112, '5': 8886, '2': 2151, '3': 2174, '1': 576, '6': 331}
            cut_num_ = None
            if num >= 300 and (cate == 1 or cate == 6):
                if cate == 1:
                    cut_num_ = 2 * cut_num
                if cate == 6:
                    cut_num_ = 3 * cut_num
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
                if y1 - y0 > random_height:
                    top = random.randint(int(y0) + 10 - random_height, int(y0))
                if x1 - x0 > random_width:
                    left = random.randint(int(x0) + 10 - random_width, int(x0))

                # 如果顶点左侧越出边界
                if x1 + 1 - random_width < 0:
                    left = random.randint(0, int(x0))

                # 如果顶点下侧越出边界
                if y1 + 1 - random_height < 0:
                    top = random.randint(0, int(y0))
                
                # 不满足以上任何情况
                if x1 - x0 <= random_width and x1 + 1 >= random_width:
                    a, b = int(x1) + 1 - random_width, int(x0) - 1
                    # 防止 a 大于等于 b
                    while a >= b:
                        a -= 5
                    left = random.randint(a, b)

                if y1 - y0 <= random_height and y1 + 1 >= random_height:
                    a, b = int(y1) + 1 - random_height, int(y0) - 1
                    while a >= b:
                        a -= 5
                    top = random.randint(a, b)

                # 开始裁剪 正样本
                # 如果横着越出边界
                if left + random_width > width:
                    left = width - random_width

                # 竖着越出边界
                if top + random_height > height:
                    top = height - random_height

                # 判断裁剪区域和目标区域有无交集，没有交集就目标区域
                if (x1 <= left and y1 <= top) or \
                   (x0 >= left + random_width and y1 <= top) or \
                   (x1 <= left and y0 >= top + random_height) or \
                   (x0 >= left + random_width and y0 >= top + random_height):
                    print('------In cut resion, no target------')
                    if x1 <= left and y1 <= top:
                        while x1 <= left and y1 <= top:
                            left -= 5
                            top -= 5
                    elif x0 >= left + random_width and y1 <= top:
                        while x0 >= left + random_width and y1 <= top:
                            left += 5
                            top -= 5
                    elif x1 <= left and y0 >= top + random_height:
                        while x1 <= left and y0 >= top + random_height:
                            left -= 5
                            top += 5
                    elif x0 >= left + random_width and y0 >= top + random_height:
                        while x0 >= left + random_width and y0 >= top + random_height:
                            left += 5
                            top += 5

                region = image.crop((left, top, left + random_width, top + random_height))
                if ismask:
                    mask_region = mask.crop((left, top, left + random_width, top + random_height))

                    # 检验，如果只切割了背景，异常中断
                    a1 = np.array(mask_region)
                    assert len(np.unique(a1)) > 1, mask_image_path
                    mask_region.save(mask_image_path)

                # 重新缩放大小
                region = region.resize((512, 512), resample=Image.LANCZOS)
                region.save(image_path)
                print(cnt, '/', 12000)
                # 遍历完一张图片，自增
                cnt += 1

                dict_p['name'] = pic_name
                dict_p['category'] = cate
                # 保留残缺区域的相对位置
                dict_p['bbox'] = [
                    round(x0 - left, 2) * (512 / random_width), 
                    round(y0 - top, 2) * (512 / random_height), 
                    round(x1 - left, 2) * (512 / random_width), 
                    round(y1 - top, 2) * (512 / random_height)
                ]
                dict_p['image_height'] = 512
                dict_p['image_width'] = 512
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
    CUTNUM = 2
    # 目标区域的尺寸
    SCALE = 10

    # 返回要裁剪的 json 的路径
    cut_json_path = utils.select_pic(json_path=JSONPATH, sample_path=SAMPLEPATH, num=NUM, all=False)

    # 汇总
    if not os.path.exists(JSONPATHSUM):
        print('汇总 json 文件')
        sum_json(json_path=cut_json_path, json_path_sum=JSONPATHSUM)

    # 按照汇总好的数据开始切
    cut_pic(json_path=JSONPATHSUM, save_path=SAVEPATH, num=NUM, scale=SCALE, cut_num=CUTNUM, ismask=False, adaptive=True)
    print('Fucking end.')
# 157001