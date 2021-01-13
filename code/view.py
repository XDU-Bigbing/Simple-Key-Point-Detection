import json
import random
from PIL import Image, ImageDraw
from collections import defaultdict

# 随机选 1000 张图片
random.seed(10)
data = set()
for i in range(1000):
    data.add(random.randint(0, 15229))

d = defaultdict(set)
JSONPATH = '../data/tile_round1_train_20201231/train_annos.json'
DATAPATH = "../data/tile_round1_train_20201231/train_imgs/"

cnt = 0
with open (JSONPATH, 'r') as f:
    dict_ = json.load(f)
    for i in data:
        if len(d[str(dict_[i]['category'])]) < 10:
            d[str(dict_[i]['category'])].add(dict_[i]['name'])
            with Image.open(DATAPATH + dict_[i]['name']) as im:
                draw = ImageDraw.Draw(im)
                area = dict_[i]['bbox']
                draw.rectangle((area[0]-2, 
                                area[1]-2, 
                                area[2]+2, 
                                area[3]+2), 
                                outline="#FF0000", 
                                width=5)
                cnt += 1
                # write to stdout
                im.save('view/' + str(dict_[i]['category'])
                 + '_' + dict_[i]['name'])
                print(cnt)

for key, value in d.items():
    print(key, len(value))
