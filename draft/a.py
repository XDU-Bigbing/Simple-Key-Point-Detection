# train_scale, test_scale, valid_scale = 0.6, 0.2, 0.1
# sum_str = "The sum number of train, test, valid must equal to 1"
# assert train_scale + test_scale + valid_scale == 1.0, sum_str

import json

JSONPATH = '../data/tile_round1_train_20201231/train_annos.json'

data_dict = {}
x = []
y = []
size = []
box = []
batchsz = 16

with open(JSONPATH, 'r') as f:
    data_dict = json.load(f)
    length = len(data_dict)
    print(length, batchsz)
    for i in range(0, length, batchsz):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        tmp4 = []
        if i + batchsz >= length:
            batchsz = i + batchsz - length
        for j in range(batchsz):
            tmp1.append(data_dict[i+j]['name'])
            print(type(data_dict[i+j]['category']))
            tmp2.append(data_dict[i+j]['category'])
            tmp3.append([data_dict[i+j]['image_height'],
                            data_dict[i+j]['image_width']])
            tmp4.append(data_dict[i+j]['bbox'])
        
        x.append(tmp1)
        y.append(tmp2)
        size.append(tmp3)
        box.append(tmp4)

print(len(x[-2]))

# class data(object):
#     def __init__(self):
#         self.data = [i for i in range(100)]
#     def __getitem__(self, index):
#         return self.data[index]

# a = data()
# for a, b in enumerate(a, 2):
#     print(a, b)
