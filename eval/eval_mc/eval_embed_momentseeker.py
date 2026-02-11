import torch
import os
import json
import torch.nn.functional as F
from tqdm import tqdm

res_dir = ""

files = os.listdir(res_dir)
bin_files = [os.path.join(res_dir, f) for f in files if f.endswith(".bin")]

res = []
for bin_f in bin_files:
    res_f = torch.load(bin_f)
    res += res_f

print(len(res))

query_dic = {}
pos_dic = {}
neg_dic = {}
for item in res:
    if item["prompt"][0]["iid"] not in neg_dic:
       neg_dic[item["prompt"][0]["iid"]] = []

    if item["prompt"][0]["type"] == "query":
        query_dic[item["prompt"][0]["iid"]] = item
    elif item["prompt"][0]["type"] == "pos":
        pos_dic[item["prompt"][0]["iid"]] = item
    else:
        neg_dic[item["prompt"][0]["iid"]].append(item)

correct = 0
total = 0

for key in tqdm(list(query_dic.keys())):
    text_embeds = query_dic[key]["mllm_embeds"]
    pos = pos_dic[key]["mllm_embeds"]
    neg_lst = [it["mllm_embeds"] for it in neg_dic[key]]
    
    mllm_embeds = [pos] + neg_lst
    mllm_embeds = torch.cat(mllm_embeds, dim=0)


    mllm_embeds = F.normalize(mllm_embeds).cuda()
    text_embeds = F.normalize(text_embeds).cuda()

    sim = text_embeds @ mllm_embeds.t()
    sim = sim.squeeze(0)

    top1 = torch.argmax(sim)
    if top1 == 0:
        correct += 1
    total += 1


accuracy = correct / total
print(f"{correct} / {total} = {accuracy:.4f}")