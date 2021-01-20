import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    '''
    t 是一组变换的操作，是列表里的元素
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    '''
    随机翻转
    image： torch.Tensor 类型
    target：字典类型
    '''
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        # 概率小于 0.5（指定参数） 时
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            # torch 的左右反转
            image = image.flip(-1)
            bbox = target["boxes"]
            # 左右翻转后，纵坐标不受影响，需要变换横坐标
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            # masks 模式下，翻转
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
