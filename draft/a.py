# train_scale, test_scale, valid_scale = 0.6, 0.2, 0.1
# sum_str = "The sum number of train, test, valid must equal to 1"
# assert train_scale + test_scale + valid_scale == 1.0, sum_str

# import json

# JSONPATH = '../data/tile_round1_train_20201231/train_annos.json'

# data_dict = {}
# x = []
# y = []
# size = []
# box = []
# batchsz = 16

# with open(JSONPATH, 'r') as f:
#     data_dict = json.load(f)
#     length = len(data_dict)
#     print(length, batchsz)
#     for i in range(0, length, batchsz):
#         tmp1 = []
#         tmp2 = []
#         tmp3 = []
#         tmp4 = []
#         if i + batchsz >= length:
#             batchsz = i + batchsz - length
#         for j in range(batchsz):
#             tmp1.append(data_dict[i+j]['name'])
#             print(type(data_dict[i+j]['category']))
#             tmp2.append(data_dict[i+j]['category'])
#             tmp3.append([data_dict[i+j]['image_height'],
#                             data_dict[i+j]['image_width']])
#             tmp4.append(data_dict[i+j]['bbox'])
        
#         x.append(tmp1)
#         y.append(tmp2)
#         size.append(tmp3)
#         box.append(tmp4)

# print(len(x[-2]))

# class data(object):
#     def __init__(self):
#         self.data = [i for i in range(100)]
#     def __getitem__(self, index):
#         return self.data[index]

# a = data()
# for a, b in enumerate(a, 2):
#     print(a, b)


# for a, b in enumerate(range(3, 7)):
#     print(a, b, sep=', ')



# def change(ls):
#     ls[1] = 16

# if __name__ == "__main__":
#     ls = [1, 2, 3]
#     change(ls)
#     print(ls)

# import json
# dict1 = {'a':1, 'b':2, 'c':3}
# dict2 = {'A':1, 'B':2, 'C':3}
# dict3 = {'q':1, 'b':2, 'z':3}
# dict_list = []
# dict_list.append(dict1)
# dict_list.append(dict2)
# dict_list.append(dict3)
# # 其实，我完全可以这样写dict_list = [dict1，dict2，dict3]；之所如上述那么些，是让大家知道，我们是动态控制的
# with open('demo3.json', mode='w', encoding='utf-8') as f:
#     # 将字典列表存入json文件中
#     json.dump(dict_list, f, indent=4)

# d = {}
# d['a'] = [1, 2, 3 ,4]
# print(d)

# import json

# path = '../data/train_annos.json'

# max_x, max_y = 512, 512
# ave_x, ave_y = 0, 0
# cnt_x, cnt_y = 0, 0
# min_x, min_y = 512, 512

# with open(path, 'r') as f:
#     data_dict = json.load(f)
#     for i in data_dict:
#         # if i['bbox'][0] >= i['bbox'][2]:
#         #     print('x error', i['name'], sep=', ')
#         # if i['bbox'][1] >= i['bbox'][3]:
#         #     print('y error', i['name'], sep=', ')
#         if i['bbox'][3] - i['bbox'][1] >= max_y:
#             max_y = i['bbox'][3] - i['bbox'][1]
#         if i['bbox'][3] - i['bbox'][1] < min_y:
#             min_y = i['bbox'][3] - i['bbox'][1]
#         if i['bbox'][2] - i['bbox'][0] >= max_x:
#             max_x = i['bbox'][2] - i['bbox'][0]
#         if i['bbox'][2] - i['bbox'][0] < min_x:
#             min_x = i['bbox'][2] - i['bbox'][0]

# print(max_x, max_y, min_x, min_y, sep=', ')

# x huge, 220_67_t20201124133440453_CAM3.jpg
# x huge, 233_116_t20201127105036415_CAM2.jpg
# x huge, 220_67_t20201124133440953_CAM1.jpg 
# y huge, 220_67_t20201124133440953_CAM1.jpg 
# y huge, 220_67_t20201124133440953_CAM1.jpg 
# x huge, 220_67_t20201124133440953_CAM1.jpg 
# y huge, 245_22_t20201128140641832_CAM1.jpg 
# y huge, 245_24_t2020112814071352_CAM2.jpg  
# y huge, 220_67_t20201124133441107_CAM2.jpg 
# x huge, 220_67_t20201124133441107_CAM2.jpg 
# y huge, 233_152_t20201127110215535_CAM2.jpg
# y huge, 241_29_t20201128123614759_CAM2.jpg 
# x huge, 254_78_t20201130152314246_CAM1.jpg 
# x huge, 220_65_t20201124133428897_CAM1.jpg
# x huge, 220_65_t2020112413342951_CAM2.jpg
# y huge, 241_29_t20201128123614611_CAM1.jpg
# y huge, 230_189_t20201126151119779_CAM2.jpg
# y huge, 230_189_t20201126151119622_CAM1.jpg
# x huge, 230_189_t20201126151119622_CAM1.jpg
# x huge, 233_149_t20201127110145219_CAM1.jpg
# x huge, 233_149_t20201127110145219_CAM1.jpg
# x huge, 254_78_t20201130152313741_CAM3.jpg
# x huge, 254_78_t20201130152314389_CAM2.jpg
# y huge, 230_167_t20201126150527680_CAM1.jpg
# x huge, 237_110_t20201127134556548_CAM2.jpg
# y huge, 245_22_t20201128140641985_CAM2.jpg
# y huge, 230_167_t20201126150527833_CAM2.jpg
# y huge, 245_24_t20201128140712900_CAM1.jpg
# x huge, 239_127_t20201128092704783_CAM2.jpg
# x huge, 220_16_t2020112413192698_CAM2.jpg
# x huge, 220_16_t20201124131925945_CAM1.jpg
# x huge, 233_116_t20201127105036261_CAM1.jpg
# y huge, 245_87_t20201128142225723_CAM2.jpg
# x huge, 220_65_t20201124133428398_CAM3.jpg
# x huge, 253_40_t20201130123650449_CAM1.jpg
# x huge, 253_40_t20201130123649947_CAM3.jpg
# x huge, 239_127_t20201128092704629_CAM1.jpg
# x huge, 233_149_t20201127110145380_CAM2.jpg
# x huge, 233_149_t20201127110145380_CAM2.jpg
# y huge, 245_87_t20201128142225571_CAM1.jpg
# x huge, 253_40_t20201130123650600_CAM2.jpg
# y huge, 245_23_t20201128140657791_CAM2.jpg

# import json

# d = {}

# with open('instances_val2017.json', 'r') as f:
#     d = json.load(f)

# with open('instances_val2017.json', 'w') as f:
#     json.dump(d, f, indent=4)

# d = {}
# print(d['a'])

# a = b = 'asdasd'

# print(a, b)

# def t():
#     return 1

# t()

# a = [1, 2, 3]
# b = [3, 4, 5]

# print(a < b)

# from operator import itemgetter
# from itertools import groupby


# # 创建记录
# records = [
#     {'address': '5412 N CLARK', 'date': '07/01/2012'},
#     {'address': '5148 N CLARK', 'date': '07/04/2012'},
#     {'address': '5800 E 58TH', 'date': '07/02/2012'},
#     {'address': '2122 N CLARK', 'date': '07/03/2012'},
#     {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
#     {'address': '1060 W ADDISON', 'date': '07/02/2012'},
#     {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
#     {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
# ]

# # 先按照给定字段将数据排序
# records.sort(key=itemgetter('date'))

# # 分组，一次迭代查找连续的相同值，返回一个值和一个迭代器对象
# for date, items in groupby(records, key=itemgetter('date')):
#     for i in items:
#         print(' ', i['address'], i['date'])
#     for i in items:
#         print('----')

# import json
# d = [
#     {
#         'name': '123',
#         'width': 12
#     },
#     {
#         'name': '12',
#         'width': 56
#     }
# ]

# with open('a.json', 'w') as f:
#     json.dump(d, f, indent=4)

# for i in range(10):
#     with open('a.json', 'r') as f:
#         d = json.load(f)
#         print(d)

# from PIL import Image
# import numpy as np

# colors = {
#     'background': '#000000',
#     '1': '#110000',
#     '2': '#002200',
#     '3': '#000033',
#     '4': '#445500',
#     '5': '#660077',
#     '6': '#008899',
# }


# im = Image.new('RGB', (3, 3), colors['5'])
# # im1 = Image.new('RGB', (2, 2), colors['6'])
# # im.paste(im1, (0, 0))
# im.save('1.png')
# mask = Image.open('1.png')
# mask = np.array(mask)
# print(mask)
# obj_ids = np.unique(mask)
# for i in obj_ids:
#     print(str(i))
# print(obj_ids, sep='  ')

    # print(value, mask, sep=', ')
# print(int('66', 16))

# d = {
#     '17': 1,
#     '34': 2,
#     '51': 3,
#     '68': 4,
#     '85': 4,
#     '102': 5,
#     '119': 5,
#     '136': 6,
#     '153': 6
# }

# for i in range(11, 110, 11):
#     print(int(str(i), 16))

# a = 'asdasdihbaesd.jpg'
# print(a[0:-4])

# import json
# d = {}
# a = set()
# with open('../data/train_annos.json') as f:
#     d = json.load(f)
#     print(len(d))
# for i in d:
#     a.add(i['name'])
# print(len(a))
# d1 = {}
# with open('../data/train_annos.json') as f:
#     d1 = json.load(f)

# print(d == d1)

# import json
# import pandas as pd

# df = pd.read_json('a.json')
# df.to_csv('a.csv', index = None)
