import pandas as pd
import os
import cv2
import numpy as np

# unuse = {'CutData/train/221_22_t20201124143147440_CAM3_0_4_0'}

# dataframe = pd.read_csv('cut_data.csv')
# image_ids = dataframe["image_id"].unique()
# image_dir = '.'
# for index in range(len(image_ids)):
#     image_id = image_ids[index]
#     print(image_id)
#     image_src = os.path.join(image_dir, str(image_id)) + ".png"
#     image = cv2.imread(image_src, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


dataframe = pd.read_csv('test_data_info.csv')
image_ids = dataframe["image_id"].unique()
image_dir = '../DATA/tile_round1_testA_20201231/sub_imgs'
for index in range(len(image_ids)):
    image_id = image_ids[index]
    print(image_id)
    image_src = os.path.join(image_dir, str(image_id)) + ".jpg"
    image = cv2.imread(image_src, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)