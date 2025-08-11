import evaluate 
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
# from rouge_chinese import Rouge
import jieba
import json
from tqdm import tqdm
import os
import re
from bert_score import BERTScorer

# rouge = Rouge()
prefix = '/workspace/fengzhuoer/andrew/checkpoints'
bertscore_scorer = BERTScorer(model_type=os.path.join(prefix, 'roberta-large'), num_layers=17, lang='en')


def compute_bleu_score(solution_str: str, ground_truth_list: list, weights=(1, 0, 0, 0), format_score=0.0, score=1.0):
    """The scoring function for Infilling.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    # reference = [t for t in ground_truth.strip()]
    reference = []
    for gt in ground_truth_list:
        reference.append([t for t in gt.strip().lower()])
    infers = [t for t in solution_str.strip().lower()]
    value_score = sentence_bleu(reference, infers, weights=weights, smoothing_function=SmoothingFunction().method1)
    return value_score * score


def compute_rouge_score(solution_str: str, ground_truth: str, rouge_gram: int, format_score=0.0, score=1.0):
    """The scoring function for Infilling.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    ref_cut = ground_truth.strip()
    inf_cut = solution_str.strip()
    try:
        if rouge_gram == 1:
            value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-1']['f']
        elif rouge_gram == 2:
            value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-2']['f']
        elif rouge_gram == 'l':
            value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-l']['f']
    except ValueError as e:
        value_score = 0

    return value_score * score


def compute_meteror_score(solution_str: str, ground_truth_list: list, format_score=0.0, score=1.0):
    tokens_sol = word_tokenize(solution_str.strip())
    tokens_gro = []
    for gt in ground_truth_list:
        tokens_gro.append(word_tokenize(gt.strip()))
    results = meteor_score(references=tokens_gro, hypothesis=tokens_sol)
    return results * score


def compute_bert_score(solution_list: list, ground_truth_list: list, language='en', format_score=0.0, score=1.0):
    results = bertscore_scorer.score(cands=solution_list, refs=ground_truth_list, batch_size=32, verbose=True) # Tuple(Tensor, Tensor, Tensor)
    return (results[2] * score).detach().tolist()


def compute_exact_match_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    return score if solution_str.strip() == ground_truth.strip() else format_score

    
def f1_score(pred_set: set, label_set: set, format_score=0.0, score=1.0):
    precision = len(pred_set.intersection(label_set)) / len(pred_set) if len(pred_set) > 0 else 0
    recall = len(pred_set.intersection(label_set)) / len(label_set) if len(label_set) > 0 else 0
    if precision + recall == 0:
        return format_score
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * score
    
    
def compute_character_f1_score(solution_str: str, ground_truth: str, format_score=0.0, score=1.0):
    solution_word = solution_str.strip().split()
    solution_characters = set(solution_word)
    ground_truth_word = ground_truth.strip().split()
    ground_truth_characters = set(ground_truth_word)
    return f1_score(solution_characters, ground_truth_characters, format_score=format_score, score=score)
    
    
def compute_math_answer_correct_score(solution_str: str, ground_truth, method="flexible", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    def extract_solution(solution_str, method="strict"):
        assert method in ["strict", "flexible"]

        if method == "strict":
            # this also tests the formatting of the model
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is None:
                final_answer = None
            else:
                final_answer = solution.group(0)
                final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) == 0:
                # no reward is there is no answer
                pass
            else:
                invalid_str = ["", "."]
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer

    answer = extract_solution(solution_str=solution_str, method=method)
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


