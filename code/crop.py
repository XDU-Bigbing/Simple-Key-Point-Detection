# -*- coding: utf-8 -*-

import os
import cv2
import json
#import numpy as np
'''滑动窗口'''


def sliding_window(item_name,
                   image,
                   step_size,
                   window_size,
                   width,
                   height,
                   shrink_ratio,
                   save_path='maskdata/preprocess/test/'):
    count = 0
    info_list = []
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            if (y + window_size[1]) <= height and (
                    x + window_size[0]) <= width:  #没超出下边界，也超出下边界
                slide = image[y:y + window_size[1], x:x + window_size[0], :]
                # slide_shrink = cv2.resize(slide, window_size, interpolation=cv2.INTER_AREA)
                #slide_shrink_gray = cv2.cvtColor(slide_shrink,cv2.COLOR_BGR2GRAY)
                sub_image_name = item_name.split('.')[0] + '_' + str(
                    count) + '.jpg'
                cv2.imwrite(save_path + sub_image_name, slide)
                info_list.append({
                    'image_name':
                    item_name,
                    'sub_image_name':
                    sub_image_name,
                    'offset': [x, y],
                    'shrink_ratio': [shrink_ratio[0], shrink_ratio[1]]
                })
                count = count + 1  #count持续加1
            if (y + window_size[1]) > height and (
                    x +
                    window_size[0]) > width:  #超出右边界，但没超出下边界 或者 超出下边界，但没超出右边界
                continue
            if (y + window_size[1]) > height and (
                    x + window_size[0]) <= width:  #超出下边界，也超出下边界
                break
    return info_list


def crop(path, item_name, step_ratio=0.7, window_size=[512, 512]):
    step_size = (int(round(step_ratio * window_size[0])),
                 int(round(step_ratio * window_size[1])))  #步长就是0.7倍的滑窗大小
    image = cv2.imread(path + item)
    height, width = image.shape[0], image.shape[1]
    new_size = (int(round(width / step_size[0] - 2) * step_size[0]) +
                window_size[0],
                int(round(height / step_size[1] - 2) * step_size[1]) +
                window_size[1])
    img_shrink = cv2.resize(image, new_size,
                            interpolation=cv2.INTER_AREA)  #改变图像大小
    shrink_ratio = (image.shape[1] / img_shrink.shape[1],
                    image.shape[0] / img_shrink.shape[0])
    info_list = sliding_window(item_name, img_shrink, step_size, window_size,
                               img_shrink.shape[1], img_shrink.shape[0],
                               shrink_ratio)
    return info_list


if __name__ == "__main__":
    result_list = []
    path = r'maskdata/test/'  #文件路径
    filelist = os.listdir(path)  # 列举图片名
    cnt = 0
    for item in filelist:
        print(cnt)
        info_list = crop(path, item)
        result_list.extend(info_list)
        cnt += 1

    with open("info.json", 'w') as f:
        json.dump(result_list, f, indent=4)
