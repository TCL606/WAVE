import json
from tqdm import tqdm

json_file = ""

with open(json_file, 'r') as fp:
    data = json.load(fp)

cnt, total = 0, 0
for item in tqdm(data):
    pred = item["pred"].replace("<|im_end|>", "")
    ref = item["ref"]

    if pred == ref:
        cnt += 1 
    total += 1

print(f"{cnt} / {total} = {cnt / total}")