import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision
import json
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import numpy as np
import engine
import model
import pandas as pd
from torchvision import transforms as T
import config
from pprint import pprint
import utils
import dataset



log_name = 'train.log'


def writelog(file, logstr):
    result_file_open = open(file, 'a')
    result_file_open.write(logstr+'\n')
    result_file_open.close()


def get_checkpoint_state(model_path, model, optimizer, scheduler):
    # 恢复上次的训练状态
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, epoch, optimizer, scheduler


def save_checkpoint_state(path, epoch, model, optimizer, scheduler):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    torch.save(checkpoint, config.MODEL_SAVE_PATH+"_{}.pth".format(epoch))


def run():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_df = pd.read_csv(config.TRAIN_CSV_PATH)

    train_dataset = dataset.train_dataset(
        train_df,
        config.TRAIN_IMAGE_DIR,
        target=config.TARGET_COL,
        transforms=T.Compose([T.ToTensor()]),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )

    print("Data Loaders created")
    writelog(log_name, "Data Loaders created")
    start_epoch = 0
    # detector = model.create_model(config.NUM_CLASSES, backbone=config.BACKBONE)
    # anchor_generator = AnchorGenerator(sizes=((4,), (8,), (16,), (32,), (64,)),
    #                                    aspect_ratios=((0.5, 1.0, 1.5, 2.0),))
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    detector = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=config.NUM_CLASSES,
                                                                                rpn_anchor_generator=rpn_anchor_generator,
                                                                                pretrained=False)  # 默认backbone为fasterrcnn_resnet50_fpn
    # detector = get_vgg16_model_FRCNN(config.NUM_CLASSES)

    # detector.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    if config.IS_CONTINUE:
        detector, start_epoch, optimizer, lr_scheduler = get_checkpoint_state(
            config.END_MODEL_PATH, detector, optimizer, lr_scheduler)

    detector.to(device)
    print("Model loaded to device")
    writelog(log_name, "Model loaded to device")

    print("---------------- Training Started --------------")
    writelog(log_name, "---------------- Training Started --------------")
    min_loss = 10000000
    for epoch in range(start_epoch, config.EPOCHS):
        loss_value = engine.train_fn(
            train_dataloader, detector, optimizer, device, lr_scheduler)
        print("epoch = {}, Training_loss = {}".format(epoch, loss_value))
        save_checkpoint_state(
            config.MODEL_SAVE_PATH+"_{}.pth".format(epoch), epoch, detector, optimizer, lr_scheduler)
        writelog(log_name, "epoch = {}, Training_loss = {}".format(
            epoch, loss_value))
        # Set the threshold as per needs
        if loss_value < min_loss:
            min_loss = loss_value
            save_checkpoint_state(config.BEST_MODEL_PATH,
                                  epoch, detector, optimizer, lr_scheduler)
            writelog(log_name, ">>>>>>>>>>>>>>>>>>>>> save model <<<<<<<<<<<<<<<<<")

    print("-" * 25)
    writelog(log_name, "-" * 25)
    print("Model Trained and Saved to Disk")
    writelog(log_name, "Model Trained and Saved to Disk")


if __name__ == "__main__":
    run()
