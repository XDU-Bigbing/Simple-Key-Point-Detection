import torch
import torchvision
import MaskData
from torchvision.transforms import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):

    # backbone 为 RESNET 101
    model = torchvision.models.detection.maskrcnn_resnet101_fpn(pretrained=True)

    # 获取分类器的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 盒子的预测换为 fastrcnn
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取 mask 分类器的输入特征
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 换掉 mask 分类器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('=================>>>>  Training on {}'.format(device))

    # 类别数量 包括一个背景
    num_classes = 6 + 1

    # 切分数据为训练和测试
    dataset = MaskData('PennFudanPed', get_transform(train=True))
    dataset_test = MaskData('PennFudanPed', get_transform(train=False))

    # 按数据的数量生成数据索引的随机数
    indices = torch.randperm(len(dataset)).tolist()
    # 获取前 N 个作为训练集
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # 后 50 个作为测试数据
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # 加载训练数据
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4)

    # 加载测试数据
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=4)

    # 模型
    model = get_model_instance_segmentation(num_classes)
    # 模型到设备
    model.to(device)

    # 建立优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)

    # 根据 epoch 调整学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 100

    for epoch in range(num_epochs):
        # 训练一个 epoch，十次打印一次
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # 更新学习率
        lr_scheduler.step()
        # 在测试集合上评价
        evaluate(model, data_loader_test, device=device)


    print("Fucking end.")


if __name__ == "__main__":
    main()