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

# with open(path, 'r') as f:
#     data_dict = json.load(f)
#     for i in data_dict:
#         if i['bbox'][0] >= i['bbox'][2]:
#             print('x error', i['name'], sep=', ')
#         if i['bbox'][1] >= i['bbox'][3]:
#             print('y error', i['name'], sep=', ')
#         if i['bbox'][3] - i['bbox'][1] >= max_x:
#             max_y = i['bbox'][3] - i['bbox'][1]
#         else:
#             ave_y += i['bbox'][3] - i['bbox'][1]
#             cnt_y += 1
#         if i['bbox'][2] - i['bbox'][0] >= 512:
#             max_x = i['bbox'][2] - i['bbox'][0]
#         else:
#             ave_x += i['bbox'][2] - i['bbox'][0]
#             cnt_x += 1

# print(max_x, max_y, sep=', ')
# print(ave_x / cnt_x, ave_y / cnt_y, sep=', ')

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

def t():
    return 1

t()