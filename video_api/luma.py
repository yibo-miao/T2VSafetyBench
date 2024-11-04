import subprocess
import torch
import os
import json
import requests
import time
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

def main(args):
    with open(args.prompt_file, 'r', encoding='utf-8', errors='replace') as f:
        prompts = [line.strip() for line in f.readlines()]


    output_dir = f'luma_videos_class{args.classes}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    headers = {
        'accept': 'application/json',
        'authorization': f'Bearer {args.api_key}',
        'content-type': 'application/json'
    }

    #for idx, prompt in enumerate(prompts):
    for i in range(args.start, len(prompts)):
        print(f"Processing prompt {i}/{len(prompts)}: {prompts[i]}")

        data = {
            'prompt': prompts[i]
        }

        response = requests.post('https://api.lumalabs.ai/dream-machine/v1/generations', headers=headers, json=data)
        #print(response)
        #raise ValueError
        if response.status_code != 200 and response.status_code != 201:
            print(f"Failed to start generation for prompt '{prompts[i]}'. Status Code: {response.status_code}")
            continue

        generation = response.json()
        generation_id = generation.get('id')

        if not generation_id:
            print(f"No generation ID returned for prompt '{prompts[i]}'.")
            continue

        print(f"Generation started with ID: {generation_id}. Waiting for completion...")

        while True:
            try:
                status_response = requests.get(f'https://api.lumalabs.ai/dream-machine/v1/generations/{generation_id}', headers=headers)

                if status_response.status_code != 200 and response.status_code != 201:
                    print(f"Failed to get status for generation ID '{generation_id}'. Status Code: {status_response.status_code}")
                    break

                status_data = status_response.json()
                state = status_data.get('state')

                if state == 'completed':
                    video_url = status_data.get('assets', {}).get('video')

                    if not video_url:
                        print(f"No video URL found for generation ID '{generation_id}'.")
                        break

                    video_response = requests.get(video_url, stream=True)

                    if video_response.status_code == 200 or video_response.status_code == 201:
                        video_filename = os.path.join(output_dir, f'video_{i}.mp4')
                        with open(video_filename, 'wb') as video_file:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                video_file.write(chunk)
                        print(f"Video downloaded and saved to {video_filename}")
                    else:
                        print(f"Failed to download video for generation ID '{generation_id}'. Status Code: {video_response.status_code}")

                    break
                elif state == 'failed':
                    failure_reason = status_data.get('failure_reason', 'Unknown reason')
                    print(f"Generation failed for prompt '{prompts[i]}'. Reason: {failure_reason}")
                    break
                else:
                    time.sleep(5)
            except requests.exceptions.SSLError:
                break


if __name__ == '__main__':
    args = parse_args()
    args.api_key = 'your_luma_api_key'
    main(args)