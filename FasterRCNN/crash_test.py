# import joblib

# helm_dict = joblib.load('data\models\helm_dict.pkl')
# mask_dict = joblib.load('data\models\mask_dict.pkl')

# print(helm_dict)

# print(mask_dict)

# import os

# import pandas as pd
# import torch

# df = pd.read_csv("data\df_train.csv")
# # print(df.head())
# # image_id = 0
# image_id = 3
# records = df[df['image_id'] == image_id]
# print(records)
# boxes = records[['xtl', 'ytl', 'xbr', 'ybr']].values
# print(boxes)
# # # # We already have xtl ytl xbr ybr. We don't need to do this
# # # # We can do this for x y w h format of boxes.
# # # # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
# # # # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
# boxes = torch.as_tensor(boxes, dtype=torch.float32)
# # # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
# # # area = torch.as_tensor(area, dtype=torch.float32)

# # # print(boxes)

# # # For the labels for has_helmet
# # labels_act = torch.ones((records.shape[0],), dtype=torch.int64)
# # print(labels_act)

# # # For has helmet
# # labels_helmet = records['has_helmet'].values
# # labels_helmet = torch.tensor(labels_helmet, dtype=torch.int64)
# # print(labels_helmet)

# # For has_mask
# labels_mask = records["has_mask"].values
# labels_mask = torch.tensor(labels_mask, dtype=torch.int64)
# print(labels_mask)

# # print(os.listdir('data/images/'))

# # image_ids = df['image_id'].unique()
# # index = 12
# # image_id = image_ids[index]
# # print(image_id)
# # print(df.shape)
