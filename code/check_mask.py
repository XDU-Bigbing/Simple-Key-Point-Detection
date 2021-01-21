import os
import numpy as np
from PIL import Image

if __name__ == "__main__":
    cnt = 0
    for file in os.listdir('maskdata/preprocess/masks/'):
        im = Image.open('maskdata/preprocess/masks/' + file)
        a = np.unique(np.array(im))[1:]
        # 只有背景区域 裁剪失败
        if len(a) == 0:
            print(file)
            cnt += 1

    print(cnt)