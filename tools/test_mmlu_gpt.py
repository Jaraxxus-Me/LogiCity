import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
from time import sleep
import random
import openai
from utils import *
import pandas as pd
import argparse
import random
from openai import OpenAI
import ast
random.seed(0)

def call_gpt(args, prompt, client, llm_version, temperature):

    new_message = [
        {"role": "system", "content": prompt[0]},
        {"role": "system", "content": prompt[1]},
        {"role": "user", "content": prompt[2]},
    ]
    gpt_response = client.chat.completions.create(
            model=llm_version,
            messages=new_message,
            max_tokens=512,
        )
    gpt_response_message = gpt_response.choices[0].message.content
    # new_message.append({"role": "system", "content": gpt_response_message})
    # new_message.append({"role": "system", "content": "Thank you for your answer. The correct answer is B."})
    # return "Answer: B."
    return gpt_response_message


def main():

    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--data_dir', type=str, default='vis_dataset/mmlu', help='the path to the MMLU dataset')
    parser.add_argument('--split', type=str, default='test', help='the split to evaluate')
    parser.add_argument('--shots', type=int, default=3, help='the number of shots')
    args = parser.parse_args()

    client = OpenAI(api_key="xxxx")

    print("Begin the MMLU evaluation.")
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    all_subject_score_list = []
    all_counter = 0 
    all_em = 0
    for subject in subjects:
        score_list = []
        subject_em = 0

        print(f"Testing the subject: {subject}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)
        val_df = pd.read_csv(os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        # build demos as the first several "training" questions from the subject
        demos = data_from_csv_to_list(train_df)[:args.shots] 
        demos_questions = [d["question"].strip() for d in demos]
        
        # build test set as the first 1 test questions from the subject
        test_set = data_from_csv_to_list(test_df)[0:1]

        # build prompts
        prompt_demo = f"Here are {args.shots} demonstrations:\n"
        for demo in demos:
            prompt_demo += "Question: " + demo["question"] + "\n"
            answer = demo["answer"]
            prompt_demo += "Answer: " + answer.strip() + "\n\n"

        # run over test example
        for each_question in test_set:
            # if each_question["question"].strip() in demos_questions:
            #     continue
            # all_counter += 1
            
            prompt_instruct = "Now try your best to answer the following question. Your answer should strictly end with the format of single letter: \'Answer: _.\'\n"
            if len(each_question["question"].split()) > 400:
                question_string = ' '.join(each_question["question"].split()[-400 : ])
            else:
                question_string = each_question["question"]
            prompt_question = "Question: " + question_string  + "\n"
            prompt_question += "Answer:"
            final_input = [prompt_demo, prompt_instruct, prompt_question]
            print(f"prompt_demo: {prompt_demo}")
            print(f"prompt_instruct: {prompt_instruct}")
            print(f"prompt_question: {prompt_question}")

            format_wrong_times = 0
            call_gpt_flag = True
            while call_gpt_flag:
                try:
                    model_answer = call_gpt(args, final_input, client=client, llm_version="gpt-3.5-turbo-0125", temperature=0)
                    em, normalized_pred, normalized_gold = single_ans_em(model_answer, each_question["answer"])
                    if len(normalized_pred) == 1:
                        call_gpt_flag = False
                    else:
                        format_wrong_times += 1
                        time.sleep(2 ** format_wrong_times)
                        print(f"Format of the answer is wrong. Retrying in 2^{format_wrong_times} seconds...")
                except Exception as e:
                    time.sleep(2 ** format_wrong_times)
                    print(f"Encountered an error: {e}. Retrying in 2^{format_wrong_times} seconds...")
            print(f"Score: {em}")
            print("\n\n")
            all_em += em
            subject_em += em 
            score_list.append(em)

        score_list = np.array(score_list)
        acc = np.mean(score_list)
        all_subject_score_list.append(score_list)

    for each_subject, score_list in zip(subjects, all_subject_score_list):
        # count number of 0 in score_list
        num_0 = np.sum(score_list == 0)
        num_1 = np.sum(score_list == 1)
        print(f"{each_subject} subject EM: {num_1}/{len(score_list)}={num_1/len(score_list)*100}%")
    
    print("\n\n")
    print("Finish the MMLU evaluation.")
    weighted_acc = np.mean(np.concatenate(all_subject_score_list))
    print ("Overall EM: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))
    print("MMLU overall acc: {:.3f}".format(weighted_acc))

    print("\n\n")



if __name__ == '__main__':
    main()