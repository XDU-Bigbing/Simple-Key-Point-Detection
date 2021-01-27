import torch
import torchvision
import torchvision.transforms as T
import model
import config
import dataset
import numpy as np
import time
from tqdm import tqdm

__all__ = ["train_fn", "eval_fn"]


def train_fn(train_dataloader, detector, optimizer, device, scheduler=None):
    detector.train()
    for images, targets, image_ids in tqdm(train_dataloader):
        images = list(image.to(device) for image in images)
        # it's key:value for t in targets.items
        # This is the format the fasterrcnn expects for targets
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return loss_value


def eval_fn(val_dataloader, detector, device, detection_threshold=0.45):
    results = []
    detector.eval()
    with torch.no_grad():
        for images, image_ids in tqdm(val_dataloader):
            images = list(image.to(device) for image in images)

            model_time = time.time()
            outputs = detector(images)
            model_time = time.time() - model_time
            # print("Inference time taken on image_batch = {}".format(model_time))

            # outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

            for i, image in enumerate(images):
                boxes = (
                    outputs[i]["boxes"].data.cpu().numpy()
                )  # Format of the output's box is [Xmin,Ymin,Xmax,Ymax]
                scores = outputs[i]["scores"].data.cpu().numpy()
                labels = outputs[i]["labels"].data.cpu().numpy()
                # boxes = boxes[scores >= detection_threshold].astype(np.float)
                # Compare the score of output with the threshold and
                # select only those boxes whose score is greater
                # scores = scores[scores >= detection_threshold]
                # labels = labels[scores >= detection_threshold]
                image_id = image_ids[i]
                result = {  # Store the image id and boxes and scores in result dict.
                    "image_id": image_id,
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
                results.append(result)

    return results
