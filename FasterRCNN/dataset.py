import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["train_dataset","valid_dataset"]


class train_dataset(Dataset):
    def __init__(self, dataframe, image_dir, target, transforms=None):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.image_dir = image_dir
        self.transforms = transforms
        self.df = dataframe
        self.target = target

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_src = os.path.join(self.image_dir, str(image_id)) + ".png"
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        # image = cv2.imread(image_id)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Scale down the pixel values of image
        image /= 255.0

        if self.transforms is not None:  # Apply transformation
            image = self.transforms(image)

        # Else for train and validation data
        records = self.df[self.df["image_id"] == image_id]
        boxes = records[["xtl", "ytl", "xbr", "ybr"]].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # For has helmet
        # labels_helmet = records['has_helmet'].values
        # labels_helmet = torch.as_tensor(labels_helmet, dtype=torch.int64)
        # print(labels_helmet)

        # labels_mask = torch.ones((records.shape[0],), dtype=torch.int64)

        # For has_mask
        labels = records[self.target].values
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # print(labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area

        return image, target, image_id


class valid_dataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_src = os.path.join(self.image_dir, str(image_id)) + ".jpg"
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Scale down the pixel values of image
        image /= 255.0

        if self.transforms is not None:  # Apply transformation
            image = self.transforms(image)


        return image, image_id
