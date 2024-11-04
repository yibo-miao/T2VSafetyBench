import requests
import json
import os
import time
from tqdm import tqdm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--api-key", default=None, type=str, help="path to save generated samples")
    parser.add_argument("--prompt-file", default=None, type=str, help="path to prompt txt file")

    parser.add_argument("--classes", default=1, type=int, help="the class number")

    parser.add_argument("--start", default=0, type=int, help="the start prompt")
    return parser.parse_args()




def downloadByUrl(url, downloadPath, fileName):
    os.makedirs(downloadPath, exist_ok=True)
    dst = os.path.join(downloadPath, fileName)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    pbar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=fileName)
    with open(dst, 'wb') as f:
        for data in response.iter_content(block_size):
            pbar.update(len(data))
            f.write(data)
    pbar.close()
    if total_size != 0 and pbar.n != total_size:
        print("ERROR, something went wrong downloading the file")
    return total_size


def main(args):


    with open(args.prompt_file, 'r', encoding='utf-8', errors='replace') as f:
        prompts = [line.strip() for line in f if line.strip()]

    address = 'https://api.pikapikapika.io/web/generate'

    headers = {
        'Authorization': f'Bearer {args.api_key}',
        'Content-Type': 'application/json',
        'User-Agent': 'PostmanRuntime/7.37.0',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    headers2 = headers.copy()
    headers2.pop('Content-Type', None)

    #for prompt in prompts:
    for i in range(args.start, len(prompts)):
        print(f"Processing prompt: {prompts[i]}")
        data = {
            "promptText": prompts[i],
            "options": {
                "aspectRatio": "5:2",
                "frameRate": 20,
                "camera": {
                    "pan": "right",
                    "tilt": "up",
                    "rotate": "cw",
                    "zoom": "in"
                },
                "parameters": {
                    "guidanceScale": 16,
                    "motion": 2,
                    "negativePrompt": "ugly",
                    "seed": 144124
                },
                "extend": False
            }
        }
        data_json = json.dumps(data)

        resp = requests.post(url=address, headers=headers, data=data_json)

        if resp.status_code != 200:
            print(f"Failed to generate video for prompt: {prompts[i]}")
            print(f"Response code: {resp.status_code}, response text: {resp.text}")
            continue

        resp_json = resp.json()
        job_id = resp_json['job']['id']
        print(f"Job ID: {job_id}")

        video_url = None
        while True:
            address2 = f'https://api.pikapikapika.io/web/jobs/{job_id}'
            try:
                resp2 = requests.get(url=address2, headers=headers2)
                if resp2.status_code != 200:
                    print(f"Failed to get job status for job_id: {job_id}")
                    break

                resp2_json = resp2.json()

                if 'videos' in resp2_json and len(resp2_json['videos']) > 0:
                    status = resp2_json['videos'][0]['status']
                    if status == 'finished':
                        video_url = resp2_json['videos'][0]['resultUrl']
                        print(f"Video ready: {video_url}")
                        break
                    elif status == 'failed':
                        print(f"Job {job_id} failed.")
                        break
                    else:
                        print(f"Job {job_id} status: {status}. Waiting...")
                else:
                    print(f"No videos available yet for job_id: {job_id}. Waiting...")

                time.sleep(5)

            except:
                continue

        if video_url:
            filename = f"{args.classes}-{i+1}.mp4"
            downloadByUrl(video_url, downloadPath=f'./pika/video', fileName=filename)
            print(f"Video saved as: f'./pika/video'/{filename}")
        else:
            print(f"Video not available for prompt: {prompts[i]}")


if __name__ == '__main__':
    args = parse_args()
    args.api_key = 'your_pika_api_key'
    main(args)