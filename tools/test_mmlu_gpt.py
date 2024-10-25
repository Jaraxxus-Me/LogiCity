import logging
import os
import pickle as pkl
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
from time import sleep
import random
from tqdm import tqdm
from utils import *
import pandas as pd
import argparse
import random
from openai import OpenAI
random.seed(0)

def call_gpt(args, client, prompt, llm_version, temperature):
    new_message = [
        {"role": "system", "content": prompt[0]},
        {"role": "system", "content": prompt[1]},
        {"role": "user", "content": prompt[2]},
    ]
    gpt_response = client.chat.completions.create(
            model=llm_version,
            messages=new_message,
            max_tokens=512,
            temperature=temperature
        )
    gpt_response_message = gpt_response.choices[0].message.content
    return gpt_response_message

def main():
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--data_dir', type=str, default='vis_dataset/mmlu_logicity/hard', help='the path to the MMLU dataset')
    parser.add_argument('--human_train_pkl', type=str, default='vis_dataset/mmlu_logicity_human/hard_train/all_train.pkl', help='the split to evaluate')
    parser.add_argument('--human_test_pkl', type=str, default='vis_dataset/mmlu_logicity_human/hard_test/all_answered.pkl', help='the split to evaluate')
    parser.add_argument('--split', type=str, default='test', help='the split to evaluate')
    parser.add_argument('--shots', type=int, default=5, help='the number of shots')
    parser.add_argument('--exp', type=str, default='gpt-4o', help='exp name')
    parser.add_argument('--good_prompt', action='store_true', help='whether to use good prompt')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=-1, help='end index')
    args = parser.parse_args()

    client = OpenAI(
    api_key='sk-BjKEtLx2h0v0Z93EV2Myju-oijzTN9DOBZDQj4Ep5rT3BlbkFJ_voUis6vY08PtIoB-scZRDgfyLJQNLh7kN5CNt23wA',  # this is also the default, it can be omitted
    )
    # Setting up the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('{}_{}_{}.log'.format(args.exp, args.start, args.end))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info("Begin the MMLU evaluation.")
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    gt_list = {}
    res_list = {}
    all_subject_score_list = []
    all_counter = 0 
    all_em = 0
    for subject in subjects:
        score_list = []
        subject_em = 0

        logger.info(f"Testing the subject: {subject}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        # build demos as the first several "training" questions from the subject
        if args.good_prompt:
            assert os.path.exists(os.path.join(args.data_dir, "good_prompt_id_{}_human.npy".format(args.shots))), "good prompt file not found"
            good_prompt_id = np.load(os.path.join(args.data_dir, "good_prompt_id_{}_human.npy".format(args.shots)))
            demos = data_from_csv_to_list(train_df)
            demos = [demos[i] for i in good_prompt_id]
        else:
            all_data = data_from_csv_to_list(train_df)
            if os.path.exists(args.human_train_pkl):
                logging.info("Loading human training data... will use it as promting examples.")
                with open(args.human_train_pkl, "rb") as f:
                    human_data = pkl.load(f)
            else:
                human_data = {}
            desired_num = {
                "A": 1,
                "B": 1,
                "C": 1,
                "D": 2,
            }
            answer_id = {}
            for i, each_data in enumerate(all_data):
                if human_data:
                    if str(i) not in human_data.keys():
                        # this is not used by human
                        continue
                if each_data["answer"] not in answer_id:
                    answer_id[each_data["answer"]] = []
                answer_id[each_data["answer"]].append(i)
            data_ids = []
            for each_answer in answer_id:
                data_ids.extend(random.sample(answer_id[each_answer], desired_num[each_answer]))
            np.save(os.path.join(args.data_dir, "good_prompt_id_{}_human.npy".format(args.shots)), np.array(data_ids))
            demos = data_from_csv_to_list(train_df)[:args.shots]
        demos_questions = [d["question"].strip() for d in demos]
        
        test_set = data_from_csv_to_list(test_df)
        if os.path.exists(args.human_test_pkl):
            logging.info("Loading human test data... will use it as testing examples.")
            with open(args.human_test_pkl, "rb") as f:
                human_test_data = pkl.load(f)
        final_test_set = {}
        for i, each_data in enumerate(test_set):
            if human_test_data:
                if str(i) not in human_test_data.keys():
                    # this is not used by human
                    continue
            final_test_set[i] = each_data
        # test_set = test_set[args.start:args.end]
        logger.info(f"Number of test questions: {len(final_test_set)}")

        prompt_demo = f"You are an expert in First-Order-Logic (FOL) Rule induction, the following question-answers are FOL reasoning examples. Here are {args.shots} demonstrations:\n"
        for demo in demos:
            prompt_demo += "Question: " + demo["question"] + "\n"
            answer = demo["answer"]
            prompt_demo += "Answer: " + answer.strip() + "\n\n"

        s = time.time()
        for data_id, each_question in final_test_set.items():
            prompt_instruct = "Now try your best to first identify the FOL rules from the examples above and then answer the following question. Your answer should strictly end with the format of single letter: \'Answer: _.\'\n"
            question_string = each_question["question"]
            prompt_question = "Question: " + question_string  + "\n"
            prompt_question += "Answer:"
            final_input = [prompt_demo, prompt_instruct, prompt_question]
            logger.info(f"prompt_demo: {prompt_demo}")
            logger.info(f"prompt_instruct: {prompt_instruct}")
            logger.info(f"prompt_question: {prompt_question}")
            logger.info("{} (time_elapsed: {}), current acc: {}".format(data_id, time.time()-s, np.mean(score_list)))
            # if (i+1) % 800 == 0:
            #     logger.info("Saving the results...")
            #     np.save("log_vis/gpt/gt_list_{}_{}_{}.npy".format(args.exp, args.start, i+args.start), gt_list)
            #     np.save("log_vis/gpt/res_list_{}_{}_{}.npy".format(args.exp, args.start, i+args.start), res_list)
            #     np.save("log_vis/gpt/all_subject_score_list_{}_{}_{}.npy".format(args.exp, args.start, i+args.start), all_subject_score_list)
            #     logger.info("Results saved.")

            format_wrong_times = 0
            call_gpt_flag = True
            while call_gpt_flag:
                try:
                    model_answer = call_gpt(args, client, final_input, llm_version="gpt-4o-mini", temperature=0)
                    em, normalized_pred, normalized_gold = single_ans_em(model_answer, each_question["answer"])
                    if len(normalized_pred) == 1:
                        gt_list[data_ids] = normalized_gold
                        res_list[data_ids] = normalized_pred
                        call_gpt_flag = False
                    else:
                        format_wrong_times += 1
                        time.sleep(2 ** format_wrong_times)
                        logger.info(f"Format of the answer is wrong. Retrying in 2^{format_wrong_times} seconds...")
                except Exception as e:
                    time.sleep(2 ** format_wrong_times)
                    logger.error(f"Encountered an error: {e}. Retrying in 2^{format_wrong_times} seconds...")
            logger.info(f"Score: {em}")
            logger.info("\n\n")
            all_em += em
            subject_em += em 
            score_list.append(em)
            all_counter += 1


        score_list = np.array(score_list)
        acc = np.mean(score_list)
        all_subject_score_list.append(score_list)

    for each_subject, score_list in zip(subjects, all_subject_score_list):
        num_0 = np.sum(score_list == 0)
        num_1 = np.sum(score_list == 1)
        logger.info(f"{each_subject} subject EM: {num_1}/{len(score_list)}={num_1/len(score_list)*100}%")

    with open("log_vis/gpt/gt_list_{}_human.pkl".format(args.exp), "wb") as f:
        pkl.dump(gt_list, f)
    with open("log_vis/gpt/res_list_{}_human.pkl".format(args.exp), "wb") as f:
        pkl.dump(res_list, f)

    np.save("log_vis/gpt/all_subject_score_list_{}_human.npy".format(args.exp), all_subject_score_list)
    logger.info("\n\n")
    logger.info("Finish the MMLU evaluation.")
    weighted_acc = np.mean(np.concatenate(all_subject_score_list))
    logger.info ("Overall EM: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))
    logger.info("MMLU overall acc: {:.3f}".format(weighted_acc))
    logger.info("\n\n")

if __name__ == '__main__':
    main()