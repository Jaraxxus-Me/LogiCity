import re 
import string
import collections
from collections import Counter
import nltk
from nltk.corpus import stopwords
import numpy as np
import json
stops = set(stopwords.words('english'))
puncs = list(string.punctuation)

def read_jsonl(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data
    
def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def normalize_answer(s):

    def remove_articles(text):
        result = re.sub(r'\b(an|the)\b', ' ', text)
        # print(f"the result of remove_articles is {result}")
        return result

    def white_space_fix(text):
        # print(f"the result of white_space_fix is {' '.join(text.split())}")
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(puncs)
        # print(f"the result of remove_punc is {''.join(ch for ch in text if ch not in exclude)}")
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # print(f"the result of lower is {text.lower()}")
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_pred, a_gold):

    # If two answers are the same, then the exact match is 1
    normalize_answer_a_pred = normalize_answer(a_pred)
    normalize_answer_a_gold = normalize_answer(a_gold)
    exact_match = int(normalize_answer_a_pred == normalize_answer_a_gold)
    print(f"normalized pred is {normalize_answer_a_pred}, normalized gold is {normalize_answer_a_gold}")
    return exact_match, normalize_answer_a_pred, normalize_answer_a_gold
    

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)  
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def single_ans_em(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    # if type(gold) !=list:
    #     gold = [gold]
    print(f"orginal pred is {pred}, orginal gold is {gold}")
    pred = answer_extract_textqa(pred)
    gold = answer_extract_textqa(gold)
    match_result, normalized_pred, normalized_gold = compute_exact(pred, gold)
    return match_result, normalized_pred, normalized_gold

def single_ans_f1(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) !=list:
        gold = [gold]
    pred = answer_extract_textqa(pred)
    return max(compute_f1(pred, a) for a in gold)

def get_exact_match(answers1, answers2):
    if type(answers1)==list:
        if len(answers1)==0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2)==list:
        if len(answers2)==0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (normalize_answer(answers1) == normalize_answer(answers2))


def get_f1(answers, predictions, is_equal=get_exact_match):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if a+b==0:
        return 0
    return 2*a*b/(a+b)

def answer_match_textqa(pred, ans):
    pred = answer_extract_textqa(pred)
    return normalize_answer(pred) == normalize_answer(ans)

def answer_extract_textqa(pred):
    prefix_1 = "answer is"
    prefix_2 = "answer:"
    prefix_3 = "Answer is"
    prefix_4 = "Answer:"
    # Classify whether the prefix is in the prediction
    if prefix_1 in pred or prefix_2 in pred or prefix_3 in pred or prefix_4 in pred:
        prefix = prefix_1 if prefix_1 in pred else prefix_2 if prefix_2 in pred else prefix_3 if prefix_3 in pred else prefix_4
        idx = pred.rfind(prefix)
        return pred[idx + len(prefix): idx + len(prefix)+4]
    return pred.strip()

def data_from_csv_to_list(dev_df):
    demos = []
    for i in range(len(dev_df)):
        # print(dev_df.iloc[i, 0])
        one_d = {}
        one_d["question"] = f"{dev_df.iloc[i, 0]}\nOption: (A) {dev_df.iloc[i, 1]} (B) {dev_df.iloc[i, 2]} (C) {dev_df.iloc[i, 3]} (D) {dev_df.iloc[i, 4]}"
        one_d["answer"] = dev_df.iloc[i, 5]
        demos.append(one_d)
    return demos