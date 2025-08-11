
import requests
import json
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def call_judge_model_api(history, response):
    messages = history + [{"role": "assistant", "content": response}]
    
    myrequest = requests.post(
        url="http://10.188.189.87:5000/conv_predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "conv": messages
        }).encode('utf-8')
    )

    print(myrequest.json()['score'][0])
    
    if myrequest.status_code != 200:
        raise Exception(f"API call failed with status code {myrequest.status_code}: {myrequest.text}")
    
    return myrequest.json()['score'][0]

        