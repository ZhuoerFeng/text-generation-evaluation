from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
# from rouge_chinese import Rouge
from rouge_score import rouge_scorer
import jieba
import json
from tqdm import tqdm
import os
import re
from bert_score import BERTScorer
import numpy as np
from scipy.special import erf
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from flask import Flask, request, jsonify
app = Flask(__name__)

config = BleurtConfig.from_pretrained('/workspace/fengzhuoer/andrew/checkpoints/BLEURT-20')
model = BleurtForSequenceClassification.from_pretrained(
    '/workspace/fengzhuoer/andrew/checkpoints/BLEURT-20', 
    torch_dtype=torch.bfloat16,
)
model.eval()
tokenizer = BleurtTokenizer.from_pretrained('/workspace/fengzhuoer/andrew/checkpoints/BLEURT-20')

# from -1 to 1
def compute_bleurt_score(solution_str: str, ground_truth: str, format_score=0.0, score=1.0):
    inputs = tokenizer(solution_str, ground_truth, padding='longest', return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        scores = model(**inputs).logits.flatten().tolist()
    return scores[0] if isinstance(solution_str, str) else scores


# rouge = Rouge()
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
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


def compute_rouge_score(solution_str: str, ground_truth: str, format_score=0.0, score=1.0):
    """The scoring function for Infilling.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    # ref_cut = ground_truth.strip()
    # inf_cut = solution_str.strip()
    # try:
    #     if rouge_gram == 1:
    #         value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-1']['f']
    #     elif rouge_gram == 2:
    #         value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-2']['f']
    #     elif rouge_gram == 'l':
    #         value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-l']['f']
    # except ValueError as e:
    #     value_score = 0

    # return value_score * score
    scores = scorer.score(ground_truth, solution_str)
    print(scores)
    return scores['rouge1'].fmeasure * score


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


def gsm8k_extract_solution(solution_str, method="strict"):
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


def compute_exact_match_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    method="flexible"
    answer = gsm8k_extract_solution(solution_str=solution_str, method=method)
    label = gsm8k_extract_solution(solution_str=ground_truth, method=method)

    if answer is None and label is None:
        return compute_character_f1_score(solution_str, ground_truth, format_score=format_score, score=score)
    elif answer is None and label is not None:
        return 0
    else:
        if answer == ground_truth or answer == label:
            return score
        else:
            return format_score


def gaussian_pdf(t, mu, sigma):
    """Cumulative distribution function (CDF) for Gaussian distribution."""
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def custom_function(t, T, sigma):
    """
    Piecewise function:
    - Linear: y(t) = t          for 0 <= t < 0.75*T
    - Scaled Gaussian CDF:      for 0.75*T <= t <= T
    The Gaussian segment passes through (0.75*T, 0.75*T) and (T, T).
    """
    t = np.asarray(t) # Supports scalar or vector input
    
    # Calculate Gaussian CDF at endpoints
    F_start = gaussian_pdf(0.75 * T, T, sigma)
    F_end   = gaussian_pdf(T, T, sigma)
    
    # Precompute scaling coefficient
    scale = (0.25 * T) / (F_end - F_start)
    shift = 0.75 * T
    
    # Piecewise function
    y = np.where(
        t < 0.75 * T,
        t,
        scale * (gaussian_pdf(t, T, sigma) - F_start) + shift
    )
    
    return y


def compute_length(solution_str: str, ground_truth: str, format_score=0.0, score=1.0):
    """The scoring function for length.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    solution_length = len(solution_str.strip())
    ground_truth_length = len(ground_truth.strip())
    if ground_truth_length == 0:
        return format_score
    if solution_length == 0:
        return format_score
    
    score = custom_function(
        t=solution_length,
        T=ground_truth_length,
        sigma=0.25 * ground_truth_length  # Adjust sigma as needed
    ) / ground_truth_length  # Normalize by ground truth length
    
    return score
    # step function: with 
    

@app.route('/', methods=['GET'])
def index():
    return jsonify({"msg": "OK"}), 200


@app.route('/predict_bleu_metric', methods=['POST'])
def predict_bleu():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    score = compute_bleu_score(
        solution_str=data["response"],
        ground_truth_list=[data["ground_truth"]],
        weights=(1, 0, 0, 0),
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_bleurt_metric', methods=['POST'])
def predict_bleurt():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    # 调用伪模型进行预测
    score = compute_bleurt_score(solution_str=data["response"], ground_truth=data["ground_truth"])
    
    score = score * 2 - 1  # Convert from [0, 1] to [-1, 1]

    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_bertscore_metric', methods=['POST'])
def predict_bertscore():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    # 调用伪模型进行预测
    score = compute_bert_score(
        solution_list=[data["response"]],
        ground_truth_list=[data["ground_truth"]],
        language='en',
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score[0],
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_exact_match_metric', methods=['POST'])
def predict_exact_match():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    # 调用伪模型进行预测
    score = compute_exact_match_score(
        solution_str=data["response"],
        ground_truth=data["ground_truth"],
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_meteor_metric', methods=['POST'])
def predict_meteor():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    # 调用伪模型进行预测
    score = compute_meteror_score(
        solution_str=data["response"],
        ground_truth_list=[data["ground_truth"]],
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_rouge_metric', methods=['POST'])
def predict_rouge():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "prompt" not in data or "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'prompt/response/ground_truth' field"}), 400
    
    
    # 调用伪模型进行预测
    score = compute_rouge_score(
        solution_str=data["response"],
        ground_truth=data["ground_truth"],
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "prompt": data["prompt"],
        "response": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200


@app.route('/predict_length', methods=['POST'])
def predict_length():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if "response" not in data or "ground_truth" not in data:
        return jsonify({"error": "Missing 'response/ground_truth' field"}), 400
    
    # 调用伪模型进行预测
    score = compute_length(
        solution_str=data["response"],
        ground_truth=data["ground_truth"],
        format_score=0.0,
        score=1.0
    )
    
    response = {
        "solution_str": data["response"],
        "ground_truth": data["ground_truth"],
        "score": score,
        "status": "success"
    }
    
    return jsonify(response), 200



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5098, processes=8)
    
# gunicorn -w 4 --threads 5 -b 0.0.0.0:5098 launch_metric_api:app