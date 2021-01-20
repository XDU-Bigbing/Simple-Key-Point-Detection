import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from engine import train_one_epoch, evaluate, predict
import utils
import transforms as T


# 在 3 号卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 加载普通数据集
class Dataset(object):
    def __init__(self, root, transforms):
        '''
        root: 根目录
        '''
        self.root = root
        self.transforms = transforms
        # 返回文件路径
        # os.path.join 根目录和参数的文件连接起来
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        # 颜色的 RGB 从16进制转为10进制，解析到对应的类别
        self.color_to_category = {
            '17': 1,
            '34': 2,
            '51': 3,
            '68': 4,
            '85': 4,
            '102': 5,
            '119': 5,
            '136': 6,
            '153': 6
        }

    def __getitem__(self, idx):
        # 加载数据
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 不同颜色表示不同实例 
        # 0 表示背景
        mask = Image.open(mask_path)

        mask = np.array(mask)
        masks = np.zeros(mask.shape, dtype=bool)
        # 实例被编码为不同的颜色
        obj_ids = np.unique(mask)
        # 移除不同背景色
        obj_ids = obj_ids[1:]

        # 返回一个 np 数组
        # mask == obj_ids 的部分为 true，以此来检测 box
        # [None] 在指定位置添加一个 1 的维度，此处为维度对齐

        # 获取每个图片上 mask 的数量 
        # 即有几个目标需要检测
        boxes = []
        print('obj_ids is ', obj_ids)
        for i, value in enumerate(obj_ids):
            print('here')
            pos = np.where(mask == obj_ids[i])
            masks[pos] = True
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            print(xmin, xmax, ymin, ymax)
            boxes.append([xmin, ymin, xmax, ymax])
        print(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 只有一个类，多少个目标就多少个 1 
        # 只检测目标，不区分具体是哪个目标
        # [1个batch，num_obj个盒子]
        # 写法参考：
        # https://github.com/pytorch/vision/blob/0985533eccf4adf583b4c9164d492d70a8226422/torchvision/models/detection/faster_rcnn.py#L330-L331
        labels = torch.zeros((1, len(obj_ids)), dtype=torch.int64)
        # 获取第 i 个标签的类别
        for i, value in enumerate(obj_ids):
            labels[0][i] = self.color_to_category[str(value)]
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 按照文件名生成图片的 id
        image_id = torch.tensor([idx])
        # 面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 非群
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# 加载测试数据
class PreData(object):
    def __init__(self, root):
        self.root = root
        # 返回文件路径
        self.imgs = list(sorted(os.listdir(os.path.join(root, "test"))))

    def __getitem__(self, idx):
        # 返回预处理的图片即可
        img_path = os.path.join(self.root, "test", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        return img

    def __len__(self):
        return len(self.imgs)


# 获取模型
def get_model_instance_segmentation(num_classes):
    # 加载训练好的模型
    # backbone 是 ResNet-50-FPN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取『类别』分类器的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替代预测器
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 掩码分类器的输入特征
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # 添加一个匿名类，类对输入的图像和目标做出转换
    transforms.append(T.ToTensor())
    # 训练模式下，增加图像，随机左右翻转
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 背景 + 瑕疵
    num_classes = 1 + 6
    # 同一路径下获取数据
    dataset = Dataset('maskdata/preprocess/', get_transform(train=True))
    dataset_test = Dataset('maskdata/preprocess/', get_transform(train=False))

    # randperm 在指定范围内生成随机数，每个数出现一次
    indices = torch.randperm(len(dataset)).tolist()
    # 训练数据与测试数据 7/3 开
    train_data_size = int(len(dataset) * 0.7)
    # 训练数据
    dataset = torch.utils.data.Subset(dataset, indices[:-train_data_size])
    # 测试数据
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-train_data_size:])

    # 加载数据
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True)
    print('Data Load is finished.....')

    # 建立模型
    model = get_model_instance_segmentation(num_classes)
    # 有现成的就加载
    if os.path.exists('model_tmp.pth'):
        model = model.load_state_dict(torch.load('model_tmp.pth'))
    model.to(device)
    print('Model has been built.....')

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)
    # 学习率衰减
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 200
    min_loss = 10000000
    print('Beginning to Train.....')
    for epoch in range(num_epochs):
        print('epoch: ', epoch, '====================>>>>>>>>>')
        # 开始训练 10 个 epoch 打印一次
        loss_value = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if loss_value < min_loss:
            print('=========>>>>>>>> epoch = {}, save model'.format(epoch))
            torch.save(model.state_dict(), 'model_tmp.pth')
            min_loss = loss_value
        # 更新学习率
        lr_scheduler.step()
        # 评估模型
        evaluate(model, data_loader_test, device=device)
        print('epoch: ', epoch, 'loss is ', loss_value)

    # 最后在保存一次模型,不覆盖训练期间保存的模型
    print("The Fucking end!")
    torch.save(model.state_dict(), 'model.pth')

    # 预测数据
    # dataset_pre = PreData('maskdata/preprocess/')
    # data_loader_pre = torch.utils.data.DataLoader(
    #     dataset_pre, batch_size=1, shuffle=False)
    # predict(model, data_loader_pre, device=device)


if __name__ == "__main__":
    main()
