import numpy as np
import pickle as pkl
import os

jessica_pkl = 'vis_dataset/mmlu_logicity_human/hard_test/all_answered.pkl'
exp_name = 'gpt-4o-mini'
gt_path = 'log_vis/gpt_cm/gt_list_{}_human.pkl'.format(exp_name)
gpt_path = 'log_vis/gpt_cm/res_list_{}_human.pkl'.format(exp_name)
use_random = True

with open(gt_path, "rb") as f:
    gt_list = pkl.load(f)

with open(gpt_path, "rb") as f:
    gpt_res_list = pkl.load(f)


with open(jessica_pkl, "rb") as f:
    jessica_data = pkl.load(f)

random_ans = {}

if use_random:
    for key in gt_list.keys():
        random_ans[key] = np.random.choice(["a", "b", "c", "d"])
    gpt_res_list = random_ans

jessica_data_id = list(jessica_data.keys())
jessica_res_list = []
gpt_res_list_p = []
gt_list_p = []
for each_id in jessica_data_id:
    jessica_res_list.append(jessica_data[each_id].lower())
    assert int(each_id) in gpt_res_list.keys()
    assert int(each_id) in gt_list.keys()
    gpt_res_list_p.append(gpt_res_list[int(each_id)])
    gt_list_p.append(gt_list[int(each_id)])
# convert strings into ints
mapping = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}

inversed_mapping = {v: k for k, v in mapping.items()}

print(len(gpt_res_list_p))
gt_list_p = [mapping[each] for each in gt_list_p]
for i, res in enumerate(gpt_res_list_p):
    if res not in mapping:
        print(f"Invalid response: {i} {res}")
        gpt_res_list_p[i] = inversed_mapping[int(res)]
gpt_res_list_p = [mapping[each] for each in gpt_res_list_p]
jessica_res_list = [mapping[each] for each in jessica_res_list]

# calculate the accuracy for each choice
choice_acc = []
choice_num = []
for i in range(4):
    correct = 0
    total = 0
    for gt, res in zip(gt_list_p, gpt_res_list_p):
        if gt == i:
            total += 1
            if res == gt:
                correct += 1
    if total == 0:
        choice_acc.append(0)
        choice_num.append(0)
        continue
    choice_acc.append(correct / total)
    choice_num.append(total)
    print(f"GPT Choice {i}: {correct}/{total} = {correct/total*100:.2f}%")

# calculate the overall accuracy
# choice_acc = [24.72, 24.81, 26.52, 24.88]
# choice_num = [4155, 2882, 715, 6488]
correct = 0
for gt, res in zip(gt_list_p, gpt_res_list_p):
    if gt == res:
        correct += 1
print(f"GPT Overall: {correct}/{len(gt_list_p)} = {correct/len(gt_list_p)*100:.2f}%")

# calculate the accuracy for each choice
choice_acc = []
choice_num = []
for i in range(4):
    correct = 0
    total = 0
    for gt, res in zip(gt_list_p, jessica_res_list):
        if gt == i:
            total += 1
            if res == gt:
                correct += 1
    if total == 0:
        choice_acc.append(0)
        choice_num.append(0)
        continue
    choice_acc.append(correct / total)
    choice_num.append(total)
    print(f"Jessica Choice {i}: {correct}/{total} = {correct/total*100:.2f}%")

# calculate the overall accuracy
# choice_acc = [24.72, 24.81, 26.52, 24.88]
# choice_num = [4155, 2882, 715, 6488]
correct = 0
for gt, res in zip(gt_list_p, jessica_res_list):
    if gt == res:
        correct += 1
print(f"Jessica Overall: {correct}/{len(gt_list_p)} = {correct/len(gt_list_p)*100:.2f}%")

# calculate the weighted accuracy
# factor = 0
# weighted_acc = 0
# for i in range(4):
#     weighted_acc += choice_acc[i] / choice_num[i]
#     factor += 1 / choice_num[i]
# weighted_acc /= factor
# print(f"Weighted: {weighted_acc*100:.2f}%")