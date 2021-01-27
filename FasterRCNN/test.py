import dataset
import utils
from pprint import pprint
import config
from torchvision import transforms as T
import pandas as pd
import model
import engine
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
import results_integration as ri
import torchvision

def run():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    valid_df = pd.read_csv(config.VALIDATION_CSV_PATH)

    valid_dataset = dataset.valid_dataset(
        valid_df,
        config.VALIDATION_IMAGE_DIR,
        transforms=T.Compose([T.ToTensor()]),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )

    print("Data Loaders created")

    detector = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=config.NUM_CLASSES,
                                        pretrained=False) # 默认backbone为fasterrcnn_resnet50_fpn
    # detector.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=torch.device('cpu')))
    detector.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    detector.to(device)

    results = engine.eval_fn(
        valid_dataloader,
        detector,
        device,
        detection_threshold=config.DETECTION_THRESHOLD,
    )

    # print(results)

    ri.results_integration(results)

    # result_file = "result.json"
    # with open(result_file, 'w') as file_object:
    #     json.dump(results, file_object)

if __name__ == "__main__":
    run()
