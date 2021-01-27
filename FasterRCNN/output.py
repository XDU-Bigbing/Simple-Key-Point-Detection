import json
import math

json_data = json.load(open('output.json', 'r'))
data_len = math.ceil(len(json_data)/5 *1.9)

new_json_data = json_data[:data_len]

with open("new_output.json", 'w') as f:
    json.dump(new_json_data, f, indent=4)
