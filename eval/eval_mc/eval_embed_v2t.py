import torch
import os
import torch.nn.functional as F

res_dir = ""

files = os.listdir(res_dir)
bin_files = [os.path.join(res_dir, f) for f in files if f.endswith(".bin")]

res = []
for bin_f in bin_files:
    res_f = torch.load(bin_f)
    res += res_f


all_mllm_embeds = F.normalize(torch.cat([it["mllm_embeds"] for it in res], dim=0)) # .cuda())
all_text_embeds = F.normalize(torch.cat([it["text_embeds"] for it in res], dim=0)) # .cuda())

sims = all_mllm_embeds @ all_text_embeds.t()

_, top_indices = torch.topk(sims, k=1, dim=1)
top_indices = top_indices.squeeze()
targets = torch.arange(all_mllm_embeds.size(0)).to(top_indices.device)

correct = torch.sum(targets == top_indices)
accuracy = correct / all_mllm_embeds.size(0)

print(f"v2t: {correct.item()} / {all_mllm_embeds.size(0)} = {accuracy.item():.4f}")