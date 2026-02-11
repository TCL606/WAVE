# Copyright (2026) Tsinghua University, Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from torch.nn.functional import cosine_similarity

res_dir = ""

files = os.listdir(res_dir)
bin_files = [os.path.join(res_dir, f) for f in files if f.endswith(".bin")]

res = []
for bin_f in bin_files:
    res_f = torch.load(bin_f)
    res += res_f

map_dic = {}
for item in res:
    vid = item["video"].split("/")[-2]
    if vid not in map_dic:
        map_dic[vid] = {}
    map_dic[vid][item["video"].split("/")[-1]] = item["mllm_embeds"]

all_query_embeds = []
all_mllm_embeds = []
for key, values in map_dic.items():
    all_query_embeds.append(values['query'])
    tmp_lst = [values["positive_clip"]]
    for i in range(9):
        tmp_lst.append(values[f"negative_clip_{i}"])
    tmp_embeds = torch.cat(tmp_lst, dim=0)
    all_mllm_embeds.append(tmp_embeds)

all_query_embeds = torch.stack(all_query_embeds, dim=0)
all_mllm_embeds = torch.stack(all_mllm_embeds, dim=0)

sims = cosine_similarity(all_query_embeds, all_mllm_embeds, dim=2)

_, top_indices = torch.topk(sims, k=1, dim=1)
targets = torch.zeros(all_mllm_embeds.size(0), dtype=torch.int64).unsqueeze(1)

correct = torch.sum(targets == top_indices)
accuracy = correct / all_mllm_embeds.size(0)

print(f"{correct.item()} / {all_mllm_embeds.size(0)} = {accuracy.item():.4f}")