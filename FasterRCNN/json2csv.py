import pandas as pd
import json

# file_name = "test_data_info.json"

# # json_data = json.load(open(file_name, 'r'))
# # print(len(json_data))
# df = pd.read_json(file_name, encoding='utf-8')
# df.to_csv(file_name[:-4] + 'csv')


json_file_name = "cut_data_12000.json"
json_data = json.load(open(json_file_name, 'r'))
csv_data = []
for item in json_data:
    csv_data.append({
        "image_id": item['name'],
        "xtl":item['bbox'][0],
        "ytl":item['bbox'][1],
        "xbr":item['bbox'][2],
        "ybr":item['bbox'][3],
        "target":item['category'],
    })

print(len(csv_data))
with open("cut_data_csv_info.json", 'w') as f:
    json.dump(csv_data, f, indent=4)

df = pd.read_json("cut_data_csv_info.json", encoding='utf-8')
df.to_csv('cut_data_12000.csv')