import asyncio
import aiohttp
import json

async def fetch_rm(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.text()


async def fetch_all(urls, payloads):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_rm(session, url, data) for url, data in zip(urls, payloads)]
        results = await asyncio.gather(*tasks)
        return results
    

def fetch_api_calls(prompt: str, response: str, ground_truth: str, weights=[0, 0, 0, 0, 0, 0, 0]):
    """
    Fetch API calls for various metrics.
    
    Args:
        prompt (str): The prompt text.
        response (str): The model's response.
        ground_truth (str): The ground truth text.
        weights (list): Weights for each metric.
        
    Returns:
        dict: A dictionary containing the scores for each metric.
    """
    payloads = [
        {"prompt": prompt, "response": response, "ground_truth": ground_truth},
    ]
    
    BASE_URL = 'http://10.188.189.99:5098'
    
    urls = [
        '{}/predict_rouge_metric'.format(BASE_URL),
        '{}/predict_bleu_metric'.format(BASE_URL),
        '{}/predict_bleurt_metric'.format(BASE_URL),
        '{}/predict_length'.format(BASE_URL),
        '{}/predict_bertscore_metric'.format(BASE_URL),
        '{}/predict_meteor_metric'.format(BASE_URL),
        '{}/predict_exact_match_metric'.format(BASE_URL),
    ]
    
    sending_urls = [url for url, w in zip(urls, weights) if w != 0]
    sending_payloads = [payloads[0] for _ in sending_urls]
    results = asyncio.run(fetch_all(sending_urls, sending_payloads))
    score = []
    cnt = 0
    for w in weights:
        if w != 0:
            score.append(float(json.loads(results[cnt])['score']))
            cnt += 1
        else:
            score.append(0.0)
    return {
            "rouge": score[0],
            "bleu": score[1],
            "bleurt": score[2],
            "length": score[3],
            "bertscore": score[4],
            "meteor": score[5],
            "exact_match": score[6]
        }, sum([w * s for w, s in zip(weights, score)]) / sum(weights) if sum(weights) > 0 else 0.0


# 示例：并行请求 3 个 API
if __name__ == '__main__':
    score_dict, score = fetch_api_calls(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        ground_truth="Paris is the capital of Germany.",
        weights=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    )
    print(score_dict)
    print(score)
    