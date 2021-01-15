import json, torch, torchvision
from PIL import Image
from torch import zeros, from_numpy
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Loader(Dataset):
    
    def __init__(self):
        self._T = transforms.Compose([
            # RGB 形式打开
            lambda x: Image.open(self._data_path + x).convert('RGB'),
            # 将取值范围 [0, 255] 的 [高度 宽度 通道] 图片转换为
            # [0, 1] 的 [通道，高度，宽度] 图片 
            transforms.ToTensor(),
        ])
        self._json_path = 'cut_pic/data_json.json'
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
        y = self._y[index]
        tmp = zeros((6, 1))
        tmp[y-1, 0] = 1
        target = {}
        target['boxes'] = self._box[i]
        target["labels"] = tmp
        return x, target

    def __len__(self):
        return len(self._x_path)


if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.train()
model = model.to(device)

num_classes = 6
batchsz = 8

DATA = Loader()
db_train = DataLoader(dataset=DATA, 
                    batch_size=batchsz, 
                    shuffle=True,
                    pin_memory=True)


for epoch in range(0, 200):
    for step, batch in enumerate(db_train):
        inputs = [batch[0].to(device)]
        target = [batch[1]]
        model(inputs, target)
