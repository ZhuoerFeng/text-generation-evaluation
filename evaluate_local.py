from xinference.client import Client
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pandas as pd


from transformers import Qwen2ForCausalLM

# from megatron.core.models.huggingface.qwen_model import Qwen2ForCausalLM

from  vllm.model_executor.models import utils

PARALLEL_NUM = 1

generate_config = {
    "max_tokens": 30000,
    "temperature": 1.0
}


client = Client("http://localhost:6123")
model = client.get_model("sft-model-ep5")
res = model.chat(
    messages=[{
        "role": "user",
        "content": "你好"
    }],
    generate_config=generate_config
)

print(res)


df = pd.read_parquet('/workspace/fengzhuoer/andrew/data/ivypanda/abs2text/test.parquet')
data = []
for line in df.to_dict(orient='records'):
    data.append(line)

fout_name = 'eval-qwen-sft-ep5.jsonl'


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=5))
def call_api(messages):
    res = model.chat(
        messages=messages,
        generate_config=generate_config
    )
    print(res)
    output = res['choices'][0]['message']['content']
    # print(output)
    return output


def run_sample_and_save(comment: dict):
    # print(comment)
    messages = comment['prompt'].tolist()

    try:
        reply = call_api(messages)
    except Exception as e:
        reply =  ""
        print(e)
    
    # comment.pop('essay')
    # comment['ai-review'] = reply
    
    ouptut_res = {
        "messages": messages,
        "response": reply,
        "target": comment['extra_info']['answer'],
    }
    
    with open(fout_name, 'a') as fout:
        fout.write(json.dumps(ouptut_res, ensure_ascii=False) + '\n')
        fout.close()


run_sample_and_save_wrapper = partial(run_sample_and_save)


with ThreadPoolExecutor(PARALLEL_NUM) as executor:
    for _ in tqdm(
        executor.map(run_sample_and_save_wrapper, data), total=len(data)
    ):
        pass



