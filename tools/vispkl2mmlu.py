import numpy as np
from tqdm import tqdm
import pickle as pkl
import os
import csv
import json

pkl_path = 'vis_dataset/hard_fixed_final/test/test_hard_fixed_final.pkl'
tgt_path = 'vis_dataset/mmlu_logicity_jessica/hard/test'
csv_path = os.path.join(tgt_path, 'hard_fixed_final_mmlu.csv')
label_json_path = os.path.join(tgt_path, 'hard_fixed_final_mmlu_label.json')

question_base = "In the scene you see a total of {} entities, they are named as follows: {}. There exist the following predicates as their attributes and relations: {}. The truth value of these predicates grounded to the entities are as follows (Only the ones that are True are provided, assume the rest are False): {}. What is the next action of entity {}?"
answer_a = "Slow"
answer_b = "Normal"
answer_c = "Fast"
answer_d = "Stop"

os.makedirs(tgt_path, exist_ok=True)

def filter(data_list):
    new_data_list = []
    for data in data_list:
        step = int(data.split('_')[1][4:])
        if step > 10:
            new_data_list.append(data)
    return new_data_list

with open(pkl_path, "rb") as f:
    vis_dataset = pkl.load(f)
vis_dataset_list = filter(list(vis_dataset.keys()))

labels = []
id = 0

with open(csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for step_name in tqdm(vis_dataset_list):
        predicates = vis_dataset[step_name]["Predicate_groundings"]
        bboxes = list(vis_dataset[step_name]["Bboxes"].values())
        types = list(vis_dataset[step_name]["Types"].values())
        next_actions = list(vis_dataset[step_name]["Next_actions"].values())

        num_entities = len(next_actions)
        entity_names = [f"Entity_{i}" for i in range(num_entities)]

        predicates_names = []
        true_groundings = []

        for k, v in predicates.items():
            if k == "Sees":
                # no need to add sees
                continue
            if len(v.shape) == 1:
                predicates_names.append(f"{k} (arity: 1)")
                for i, entity_name in enumerate(entity_names):
                    if v[i] == 1:
                        true_groundings.append(f"{k}({entity_name})")
            elif len(v.shape) == 2:
                predicates_names.append(f"{k} (arity: 2)")
                for i, entity_name in enumerate(entity_names):
                    for j, entity_name2 in enumerate(entity_names):
                        if i != j and v[i][j] == 1:
                            true_groundings.append(f"{k}({entity_name}, {entity_name2})")

        for i, e in enumerate(entity_names):
            if types[i] != 'Car':
                continue
            question = question_base.format(num_entities, ", ".join(entity_names), ", ".join(predicates_names), ", ".join(true_groundings), e)
            if next_actions[i] == 0:
                answer = "A"
            elif next_actions[i] == 1:
                answer = "B"
            elif next_actions[i] == 2:
                answer = "C"
            elif next_actions[i] == 3:
                answer = "D"
            csv_writer.writerow([question, answer_a, answer_b, answer_c, answer_d, answer])

            # Collecting labels
            self_predicates = [p for p in true_groundings if f"({e})" in p or f"({e}, " in p or f", {e})" in p]
            related_entities = []
            for p in self_predicates:
                if f"({e}, " in p:
                    ent_r = p.replace(f"({e}, ", "").split(")")[0]
                    related_entities.append(ent_r)
                elif f", {e})" in p:
                    ent_r = p.replace(f", {e})", "").split("(")[1]
                    related_entities.append(ent_r)
            label_data = {
                "id": id,
                "question": question,
                "answer": answer,
                "self_predicates": self_predicates,
                "self_entity": e,
                "related_entities": related_entities
            }
            labels.append(label_data)

            id += 1

# Write labels to JSON file
with open(label_json_path, 'w') as json_file:
    json.dump(labels, json_file, indent=4)

print("Data successfully written to CSV and label JSON files in MMLU format.")
