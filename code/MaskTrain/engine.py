import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


# 主文件中的训练
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    # 设置为训练模式
    model.train()
    # 还没懂这是什么
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 添加一个 key 为 lr，值为 SmoothedValue 对象的元素
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 用途是？
    header = 'Epoch: [{}]'.format(epoch)
    # 取消学习率衰减
    lr_scheduler = None

    # 第一次循环时
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        # 设置每个 epoch 的学习率衰减
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 开始训练
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # model 传入的参数是列表
        images = list(image.to(device) for image in images)
        # 这里修改源代码, 之前的源代码只能获取 key, 但需要 value
        targets['boxes'] = targets['boxes'].reshape(-1, 4)
        targets = [{k: v.to(device) for k, v in targets.items()}]
        # 模型返回的误差，包括 分类、回归和 mask 的误差
        loss_dict = model(images, targets)
        # 对 loss 求和
        losses = sum(loss for loss in loss_dict.values())

        # 获取所有卡上的平均误差
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 平均后的三个误差求和
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 清空上次梯度
        optimizer.zero_grad()
        # 应该用平均误差去反向传播，和我的认知的 reduce_all 有所偏差
        losses.backward()
        # 优化
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # 更新的是 累积的 loss ? 下一次就没了啊
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return loss_value



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    # 在 CPU 上单线程评估
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 获取各种类别的标签
    coco = get_coco_api_from_dataset(data_loader.dataset)
    # 张量类型
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def predict(model, data_loader, device):

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    result = []
    for image in data_loader:
        image = list(img.to(device) for img in image)
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # 理论上返回 box label socres masks
        print(outputs)