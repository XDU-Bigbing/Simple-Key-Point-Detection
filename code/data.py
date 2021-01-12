import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Loader(Dataset):
    '''
    加载数据集的类
    '''
    def __init__(self, sum_num, data_path, json_path):
        '''
        sum_num   : 总共数据的数量
        data_path : 数据集的相对路径
        json_path : json 文件相对路径
        
        '''
        self._data_num = sum_num
        self._data_path = data_path
        self._json_path = json_path

        # 图片格式转换
        self._transform = transforms.Compose([
            # RGB 形式打开
            lambda x: Image.open(x).convert('RGB'),
            # 将取值范围 [0, 255] 的 [高度 宽度 通道] 图片转换为
            # [0, 1] 的 [通道，高度，宽度] 图片 
            transforms.ToTensor(),
            # 标准化到 [-1, 1]
            # transforms.Normalize(
            #     # 每个通道的均值
            #     (0.5, 0.5, 0.5),
            #     # 每个通道的方差 
            #     (0.2, 0.2, 0.2)
            # )
        ])

        # 创建数据
        self.init()

    def init(self):
        # 文件名
        self._x_path = []
        # 标签
        self._y = []        
        # 尺寸
        self._size = []
        # 位置
        self._box = []

        with open(self._json_path, 'r') as f:
            data_dict = json.load(f)
            for i in range(len(data_dict)):
                self._x_path.append(data_dict[i]['name'])
                self._y.append(data_dict[i]['category'])
                self._size.append([data_dict[i]['image_height'],
                                   data_dict[i]['image_width']])
                self._box.append(data_dict[i]['bbox'])
    
    # 访问数据
    def __getitem__(self, index):
        x = self._x_path[index]
        x = self._transform(x)
        y = torch.LongTensor(y)
        return x, y

    # 返回数据集的大小
    def __len__(self):
        return self.batchsz

    def calc_pic_normal_sigma(self):
        '''
        计算图像的均值和方差
        '''
        pass
