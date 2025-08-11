# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
import jieba
import json
from tqdm import tqdm

rouge = Rouge()


def compute_bleu_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """The scoring function for Infilling.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    reference = [t for t in ground_truth.strip()]
    infers = [t for t in solution_str.strip()]
    value_score = sentence_bleu([reference], infers, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    return value_score * score


def compute_rouge_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """The scoring function for Infilling.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    # ref_cut = ' '.join(jieba.cut(ground_truth))
    # inf_cut = ' '.join(jieba.cut(solution_str))
    ref_cut = ground_truth.strip()
    inf_cut = solution_str.strip()
    try:
        value_score = rouge.get_scores(ref_cut, inf_cut, ignore_empty=True)[0]['rouge-l']['f']
    except ValueError as e:
        value_score = 0

    return value_score * score


def eval(filename):
    fin = open(filename).readlines()
    scores = {}
    for line in tqdm(fin):
        line = json.loads(line)
        response = line['response']
        target = line['target']
        rouges = compute_rouge_score(response, target)
        bleus = compute_bleu_score(response, target)
        scores = {
            'rouge': rouges,
            'bleu': bleus
        }
    
    print(" {} : bleu: {:.3f}, rouge: {:.3f}".format(filename, scores['bleu'], scores['rouge']))
    
    
def main():
    # eval ('eval-qwen-0.5-base.jsonl')
    # eval ('eval-qwen-sft.jsonl')
    # eval ('eval-qwen-grpo-v2.jsonl')
    eval ('eval-qwen-sft-ep5.jsonl')
    
    
if __name__ == "__main__":
    main()