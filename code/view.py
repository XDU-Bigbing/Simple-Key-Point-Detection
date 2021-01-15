'''
File: view.py
Project: code
File Created: Wednesday, 13th January 2021 9:53:01 am
-----------
Last Modified: Wednesday, 13th January 2021 4:09:03 pm
Copyright 2020 - 2021 XDU, XDU BigBing
-----------
Description: 独立于项目，每个类随机选择 10 张图片，标注异常区域，观察结果
'''

import utils
from PIL import Image, ImageDraw

JSONPATH = 'data/train_annos.json'
DATAPATH = "data/trainval/"

# 进度条计数器
cnt = 0
d = utils.select_pic(json_path=JSONPATH, data_path=DATAPATH)

# d.items() ：{'1':{(1, 2, 3), (4, 5, 6)}}
for key, value in d.items():
    for i, s in enumerate(value):
        with Image.open(DATAPATH + value[i][0]) as im:
            draw = ImageDraw.Draw(im)
            area = value[i][1]
            draw.rectangle((area[0]-2, 
                            area[1]-2, 
                            area[2]+2, 
                            area[3]+2), 
                            outline="#FF0000", 
                            width=5)
            cnt += 1
            # write to stdout
            im.save('view/' + str(value[i][2])
            + '_' + value[i][0])
            print(cnt)

for key, value in d.items():
    print(key, len(value))
