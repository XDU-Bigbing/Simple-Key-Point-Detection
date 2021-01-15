'''
File: coco.py
Project: code
File Created: Friday, 15th January 2021 3:27:48 pm
Author: lanling (https://github.com/muyuuuu)
-----------
Last Modified: Friday, 15th January 2021 9:04:07 pm
Modified By: lanling (https://github.com/muyuuuu)
Copyright 2020 - 2021 XDU, XDU
-----------
Description: 制作 COCO 数据集的 json
'''

import json


def get_image_id(json_path):
    '''
    # 生成每一个图片的唯一 id 
    # 图片名相同，id 相同
    '''
    cnt = 1
    # 图片 id 的保存
    image_id = {}
    # train_data path 的路径
    with open('trainval/data_json.json', 'r') as f:
        d = json.load(f)
        for i in range(len(d)):
            if d[i]['name'] not in image_id.keys():
                image_id[d[i]['name']] = cnt
                cnt += 1

    print('image id number is {}'.format(len(image_id.keys())))
    return image_id


def make_json(image_id, cnt, json_path, data_path):
    '''
    制作 COCO 数据的 json
    '''
    with open('trainval/data_json.json', 'r') as f:
        d = json.load(f)
        # 保存的结果
        save_dict = {}
        info_dict = {}
        info_dict['version'] = 1.0
        save_dict['info'] = info_dict
        save_dict['licenses'] = 'null'

        # images 信息和 annotations 信息
        images = []
        annotations = []
        for i in range(len(d)):

            tmp = {}
            tmp['file_name'] = d[i]['name']
            tmp['id'] = image_id[d[i]['name']]
            tmp['height'] = 512
            tmp['width'] = 512
            images.append(tmp)

            tmp1 = {}
            # x1, y1, x2, y1, x2, y2, x1, y2
            tmp1['segmentation'] = [[d[i]['bbox'][0],
                                     d[i]['bbox'][1],
                                     d[i]['bbox'][2],
                                     d[i]['bbox'][1],
                                     d[i]['bbox'][2],
                                     d[i]['bbox'][3],
                                     d[i]['bbox'][0],
                                     d[i]['bbox'][3],]]
            tmp1['area'] = (d[i]['bbox'][2] - d[i]['bbox'][0]) * (d[i]['bbox'][3] - d[i]['bbox'][1])
            tmp1['iscrowd'] = 0
            tmp1['image_id'] = image_id[d[i]['name']]
            tmp1['bbox'] = d[i]['bbox']
            tmp1['category_id'] = int(d[i]['category'])
            tmp1['id'] = cnt
            annotations.append(tmp1)
            cnt += 1

        print('box number is {}'.format(cnt))

        # 类别信息 一共有 6 个类
        string = ['background', 'edge_exception', 'angular_exception', 
                  'white_dot', 'light_block', 'dark_dot', 'aperture_biemish']
        classes = []
        for i in range(6):
            category = {}
            category['name'] = category['supercategory'] = string[i]
            category['id'] = i + 1
            classes.append(category)

        # 保存结果
        save_dict['images'] = images
        save_dict['annotations'] = annotations
        save_dict['categories'] = classes
        with open(JSONPATH+'instances_trainval.json', 'w') as f:
            json.dump(save_dict, f, indent=4)


if __name__ == "__main__":
    
    DATAPATH = 'trainval/'
    JSONPATH = 'annotations/'
    # 盒子的 id 全局唯一
    cnt = 1
    image_id = get_image_id(json_path=JSONPATH)
    make_json(image_id=image_id, cnt=cnt, json_path=JSONPATH, data_path=DATAPATH)
