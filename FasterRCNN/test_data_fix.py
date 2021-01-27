import os
import json

data_dir = '../DATA/tile_round1_testA_20201231/sub_imgs'
list_dir= os.listdir(data_dir)
result = []
for i in range(len(list_dir)):
    print(list_dir[i])
    dir_path = os.path.join(data_dir, list_dir[i])
    if os.path.isdir(dir_path):
        list_file = os.listdir(dir_path)
        for j in range(len(list_file)):
            info = {
                'image_id': os.path.join(list_dir[i], list_file[j].split('.')[0])
            }
            result.append(info)

print(len(result))
with open("test_data_info.json", 'w') as f:
    json.dump(result, f, indent=4)

