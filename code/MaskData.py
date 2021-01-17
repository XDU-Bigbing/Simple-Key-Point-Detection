'''
File: MaskData.py
Project: code
File Created: Saturday, 16th January 2021 4:04:05 pm
-----------
Last Modified: Saturday, 16th January 2021 4:04:16 pm
Copyright 2020 - 2021 XDU, XDU
-----------
Description: 加载 mask 数据的类
'''

import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms


# 加载数据的类
class Dataset(object):
    def __init__(self, root):
        '''
        root: 数据的根目录
        mode: 训练模式或测试模式
        '''
        self.root = root
        # 加载数据的文件名
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        # 图片格式转换
        self._transforms = transforms.Compose([
            # RGB 形式打开
            lambda x: Image.open(x).convert('RGB'),
            # 将取值范围 [0, 255] 的 [高度 宽度 通道] 图片转换为
            # [0, 1] 的 [通道，高度，宽度] 图片 
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        # 获取 图片 与 mask 图片
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # mask 图片中 0 表示背景
        mask = Image.open(mask_path)
        # 转为 numpy 数组
        mask = np.array(mask)
        # 取出 mask 图片中所有的不同的颜色
        obj_ids = np.unique(mask)
        # 选择区域的颜色，抛弃背景 
        # 一张图片上有好几个 mask
        obj_ids = obj_ids[1:]

        # 将颜色编码的 mask 拆分为二值 masks
        masks = mask == obj_ids[:, None, None]

        # 共这么多目标
        num_objs = len(obj_ids)
        # 获取每一个 mask 的边界
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转化为 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 有多少个目标，就有多少个标签
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 图片的 id
        image_id = torch.tensor([idx])
        # 图片的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 非群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img, target = self._transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)