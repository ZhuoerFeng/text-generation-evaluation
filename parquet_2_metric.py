import pandas as pd 
import os
from utils.query_metric import compute_bleu_score, compute_meteror_score, compute_bert_score

def main():
    prefix = '/workspace/fengzhuoer/andrew/data/huggingRM/verlinference'
    df = pd.read_parquet(os.path.join(prefix, 'generation', 'all-9-task', 'qwen3-0.6b-base.parquet'))
    print(len(df))
    print(df.columns)
    # response | extra_info['answer']
    df['bleu_score'] = df.apply(lambda x: compute_bleu_score(x['responses'].item(), [x['extra_info']['answer']], format_score=0.0, score=1.0), axis=1)
    df['meteor_score'] = df.apply(lambda x: compute_meteror_score(x['responses'].item(), [x['extra_info']['answer']], format_score=0.0, score=1.0), axis=1)
    
    df_grouped = df.groupby('data_source').agg({'bleu_score': 'mean', 'meteor_score': 'mean'}).reset_index()
    df_grouped_cpy = df_grouped.copy()
    print(df_grouped_cpy.to_markdown(index=False))
    df_grouped.to_csv(os.path.join(prefix, 'generation', 'all-9-task', 'qwen3-0.6b-base.csv'), index=False)

    
if __name__ == '__main__':
    main()

