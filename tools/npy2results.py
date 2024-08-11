import numpy as np
import os

gt_list = np.load("log_vis/gpt/gt_list_gpt35t_5shot_gp.npy")
res_list = np.load("log_vis/gpt/res_list_gpt35t_5shot_gp.npy")

# convert strings into ints
mapping = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}

inversed_mapping = {v: k for k, v in mapping.items()}

print(len(res_list))
gt_list = [mapping[each] for each in gt_list]
for i, res in enumerate(res_list):
    if res not in mapping:
        print(f"Invalid response: {i} {res}")
        res_list[i] = inversed_mapping[int(res)]
res_list = [mapping[each] for each in res_list]

# calculate the accuracy for each choice
choice_acc = []
choice_num = []
for i in range(4):
    correct = 0
    total = 0
    for gt, res in zip(gt_list, res_list):
        if gt == i:
            total += 1
            if res == gt:
                correct += 1
    choice_acc.append(correct / total)
    choice_num.append(total)
    print(f"Choice {i}: {correct}/{total} = {correct/total*100:.2f}%")

# calculate the overall accuracy
choice_acc = [24.72, 24.81, 26.52, 24.88]
choice_num = [4155, 2882, 715, 6488]
correct = 0
for gt, res in zip(gt_list, res_list):
    if gt == res:
        correct += 1
print(f"Overall: {correct}/{len(gt_list)} = {correct/len(gt_list)*100:.2f}%")

# calculate the weighted accuracy
factor = 0
weighted_acc = 0
for i in range(4):
    weighted_acc += choice_acc[i] / choice_num[i]
    factor += 1 / choice_num[i]
weighted_acc /= factor
print(f"Weighted: {weighted_acc*100:.2f}%")