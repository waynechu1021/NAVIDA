import json
import os
import random
import re
import gzip
from tqdm import tqdm
import argparse

def json2file(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(len(data_list))

# r2r
def process_r2r(output_path = 'data/sub_dataset/r2r.jsonl'):
    new_data = []
    with gzip.open('data/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz') as f:
        data = f.read() # returns a byte string `b'`
    data = json.loads(data)

    with gzip.open('data/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz') as f:
        data_gt = f.read() # returns a byte string `b'`
    data_gt = json.loads(data_gt)

    for item in tqdm(data['episodes']):
        episode_id = int(item['episode_id'])
        video_id = episode_id
        assert episode_id == int(video_id), f"{episode_id} != {int(video_id)}"
        action_list = data_gt[str(episode_id)]['actions']
        instruction = item['instruction']['instruction_text']
        new_item = {
                'episode_id': episode_id,
                'video_id': video_id,
                'instruction': instruction,
                'actions': action_list
            }
        new_data.append(new_item)
    json2file(new_data, output_path)

# rxr
def process_rxr(output_path = 'data/sub_dataset/rxr.jsonl'):
    new_data = []
    with gzip.open('data/RxR_VLNCE_v0/train/train_guide.json.gz') as f:
        data = f.read() # returns a byte string `b'`
    data = json.loads(data)

    with gzip.open('data/RxR_VLNCE_v0/train/train_guide_gt.json.gz') as f:
        data_gt = f.read() # returns a byte string `b'`
    data_gt = json.loads(data_gt)

    for item in tqdm(data['episodes']):
        if item['instruction']['language'] not in ['en-US', 'en-IN']:
            continue
        episode_id = int(item['episode_id'])
        if episode_id == 50538:
            print('here')
        video_id = episode_id
        assert episode_id == int(video_id), f"{episode_id} != {int(video_id)}"
        action_list = data_gt[str(episode_id)]['actions']
        instruction = item['instruction']['instruction_text']
        new_item = {
                'episode_id': episode_id,
                'video_id': video_id,
                'instruction': instruction,
                'actions': action_list
            }
        new_data.append(new_item)
    json2file(new_data, output_path)

# r2r envdrop
def process_envdrop(output_path = 'data/sub_dataset/envdrop.jsonl'):
    new_data = []
    with gzip.open('data/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz') as f:
        data = f.read() # returns a byte string `b'`
    data = json.loads(data)

    with gzip.open('data/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop_gt.json.gz') as f:
        data_gt = f.read() # returns a byte string `b'`
    data_gt = json.loads(data_gt)

    for item in tqdm(data['episodes']):
        episode_id = int(item['episode_id'])
        video_id = episode_id
        assert episode_id == int(video_id), f"{episode_id} != {int(video_id)}"
        action_list = data_gt[str(episode_id)]['actions']
        instruction = item['instruction']['instruction_text']
        new_item = {
                'episode_id': episode_id,
                'video_id': video_id,
                'instruction': instruction,
                'actions': action_list
            }
        new_data.append(new_item)
    json2file(new_data, output_path)

# scale vln
def process_scalevln(output_path = 'data/sub_dataset/scalevln.jsonl'):
    new_data = []
    with open('data/ScaleVLN/annotations.json','r') as f:
        data = json.load(f) # returns a byte string `b'`
    for item in tqdm(data):
        video_id = item['video'].split('_')[-1]
        episode_id = item['id']
        assert episode_id == int(video_id), f"{episode_id} != {int(video_id)}"
        for instruction in item['instructions']:
            new_item = {
                'episode_id': episode_id,
                'video_id': episode_id,
                'instruction': instruction,
                'actions': item['actions'][1:] + [0]
            }
            new_data.append(new_item)
    json2file(new_data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    
    args = parser.parse_args()
    dataname2func = {
        'r2r': process_r2r,
        'rxr': process_rxr,
        'envdrop': process_envdrop,
        'scalevln': process_scalevln
    }
    print(args.dataset_name)
    for dataset_name in args.dataset_name:
        FUNC = dataname2func[dataset_name]
        FUNC()
        