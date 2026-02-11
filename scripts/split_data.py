import json
import sys
import os
import random

random.seed(2025)

if __name__ == "__main__":
    data_file = sys.argv[1]
    split_num = sys.argv[2]
    target_dir = sys.argv[3]

    with open(data_file, "r") as f:
        data = json.load(f)
    random.shuffle(data)

    per_len = len(data) // int(split_num)

    remainder = len(data) % int(split_num)

    group_counts = [per_len] * int(split_num)

    for i in range(remainder):
        group_counts[i] += 1

    os.makedirs(target_dir, exist_ok=True)

    total_count = 0

    for i in range(int(split_num) - 1):
        # data_file = f"{target_dir}/{i}.json"
        data_file = os.path.join(target_dir, f"{i}.json")
        print(data_file)
        with open(data_file, "w") as f:
            json.dump(data[total_count:total_count+group_counts[i]], f)
        print(group_counts[i])
        total_count += group_counts[i]

    data_file = os.path.join(target_dir, f"{int(split_num) - 1}.json")
    with open(data_file, "w") as f:
        json.dump(data[total_count:], f)
    print(data_file)
    print(len(data[total_count:]))
