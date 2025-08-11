import pandas as pd 
import os
from utils.query_metric import compute_bleu_score, compute_meteror_score, compute_bert_score
from utils.query_reward_model import call_judge_model_api
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# def test_judge_model_api():
#     history = [{"role": "user", "content": "Where is the capital of Japan?"}]
#     response = "Tokyo"
    
#     score = call_judge_model_api(history, response)
#     print(f"Score: {score}")


def deal_with_async_call(dataframe):
    prompts = dataframe['prompt'].tolist()
    responses = dataframe['responses'].tolist()
    
    data = []
    for p, r in zip(prompts, responses):
        item = {
            'prompt': p.tolist(),
            'response': r.item(),
        }
        data.append(item)
        
    def process_item(item: dict):
        score = call_judge_model_api(item['prompt'], item['response'])
        item['score'] = score
        return item
    
    partial_process_item = partial(process_item)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(partial_process_item, data), total=len(data), desc="Processing items"))

    return results


def main():
    prefix = '/workspace/fengzhuoer/andrew/data/huggingRM/verlinference'
    df = pd.read_parquet(os.path.join(prefix, 'generation', 'all-9-task', 'qwen3-0.6b-sft.parquet'))
    print(len(df))
    print(df.columns)
    # response | extra_info['answer']
    
    df['bleu_score'] = df.apply(lambda x: compute_bleu_score(x['responses'].item(), [x['extra_info']['answer']], format_score=0.0, score=1.0), axis=1)
    df['meteor_score'] = df.apply(lambda x: compute_meteror_score(x['responses'].item(), [x['extra_info']['answer']], format_score=0.0, score=1.0), axis=1)
    rm_results = deal_with_async_call(df)
    df['rm_score'] = [item['score'] for item in rm_results]
    
    df_grouped = df.groupby('data_source').agg({'bleu_score': 'mean', 'meteor_score': 'mean', 'rm_score': 'mean'}).reset_index()

    df_grouped_cpy = df_grouped.copy()
    print(df_grouped_cpy.to_markdown(index=False))
    df_grouped.to_csv(os.path.join(prefix, 'generation', 'all-9-task', 'qwen3-0.6b-base.csv'), index=False)

    
if __name__ == '__main__':
    main()
