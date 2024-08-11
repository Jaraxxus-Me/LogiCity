import logging
import os
import argparse
import numpy as np
import time
import random
from utils import *
import pandas as pd
import argparse
import random
random.seed(0)

def main():
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--data_dir', type=str, default='vis_dataset/mmlu_logicity/hard', help='the path to the MMLU dataset')
    parser.add_argument('--split', type=str, default='test', help='the split to evaluate')
    parser.add_argument('--shots', type=int, default=5, help='the number of shots')
    parser.add_argument('--exp', type=str, default='random5', help='exp name')
    parser.add_argument('--good_prompt', action='store_true', help='whether to use good prompt')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=-1, help='end index')
    args = parser.parse_args()

    random.seed(4)
    # Setting up the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('{}_{}_{}.log'.format(args.exp, args.start, args.end))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info("Begin the MMLU evaluation.")
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    gt_list = []
    res_list = []
    all_subject_score_list = []
    all_counter = 0 
    all_em = 0
    for subject in subjects:
        score_list = []
        subject_em = 0

        logger.info(f"Testing the subject: {subject}")
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
        
        test_set = data_from_csv_to_list(test_df)
        test_set = test_set[args.start:args.end]
        logger.info(f"Number of test questions: {len(test_set)}")

        s = time.time()
        for i, each_question in enumerate(test_set):
            logger.info("{}/{} (time_elapsed: {}), current acc: {}".format(i, len(test_set), time.time()-s, np.mean(score_list)))
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
                    model_answer = random.choice(["A", "B", "C", "D"])
                    em, normalized_pred, normalized_gold = single_ans_em(model_answer, each_question["answer"])
                    if len(normalized_pred) == 1:
                        gt_list.append(normalized_gold)
                        res_list.append(normalized_pred)
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
        all_subject_score_list.append(score_list)

    for each_subject, score_list in zip(subjects, all_subject_score_list):
        num_0 = np.sum(score_list == 0)
        num_1 = np.sum(score_list == 1)
        logger.info(f"{each_subject} subject EM: {num_1}/{len(score_list)}={num_1/len(score_list)*100}%")

    np.save("log_vis/gpt/gt_list_{}_{}_{}.npy".format(args.exp, args.start, args.end), gt_list)
    np.save("log_vis/gpt/res_list_{}_{}_{}.npy".format(args.exp, args.start, args.end), res_list)
    np.save("log_vis/gpt/all_subject_score_list_{}_{}_{}.npy".format(args.exp, args.start, args.end), all_subject_score_list)
    logger.info("\n\n")
    logger.info("Finish the MMLU evaluation.")
    weighted_acc = np.mean(np.concatenate(all_subject_score_list))
    logger.info ("Overall EM: {}/{}={}%".format(all_em, all_counter, (all_em / all_counter) * 100))
    logger.info("MMLU overall acc: {:.3f}".format(weighted_acc))
    logger.info("\n\n")

if __name__ == '__main__':
    main()