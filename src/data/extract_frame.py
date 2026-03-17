import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ['GLOG_minloglevel'] = '2'
import json
import PIL.Image as Image
from tqdm import tqdm
import multiprocessing as mp
import time
import copy
import numpy as np
import habitat
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import argparse
from habitat_extensions import measures, task

CONFIG = {
    'r2r': {
        'config_path' : "./config/vln_r2r_train.yaml",
        'image_path': 'data/images/r2r',
        'annotation_path': 'data/sub_dataset/r2r.jsonl',
    },
    'rxr': {
        'config_path' : "./config/vln_rxr_train.yaml",
        'image_path': 'data/images/rxr',
        'annotation_path': 'data/sub_dataset/rxr.jsonl',
    },
    'envdrop': {
        'config_path' : "./config/vln_envdrop.yaml",
        'image_path': 'data/images/envdrop',
        'annotation_path': 'data/sub_dataset/envdrop.jsonl',
    },
    'scalevln': {
        'config_path' : "./config/vln_scalevln.yaml",
        'image_path': 'data/images/scalevln',
        'annotation_path': 'data/sub_dataset/scalevln.jsonl',
    },
}


def extract_data(
        result_queue, 
        env_config, 
        annotations, 
        dataset, 
        save_image = False,
        image_path = None
    ):
    env = habitat.Env(env_config.habitat, dataset)
    if save_image:
        os.makedirs(image_path, exist_ok=True)

    assert len(annotations) == len(env.episodes)
    print(f"total episode = {len(annotations)}")

    for idx, episode in enumerate(env.episodes):
        output_dict = {
            "time_per_episode": 0,
        }
        episode_start_time = time.time()
        env.current_episode = episode
        observation = env.reset()
        annotation = annotations[idx]
        assert annotation['episode_id'] == int(episode.episode_id)
        reference_actions = annotation["actions"]
        step_id = 0 
        curren_image_path = os.path.join(image_path,str(episode.episode_id))
        os.makedirs(curren_image_path, exist_ok=True)
        while not env.episode_over:
            rgb = observation["rgb"]
            rgb_frame = Image.fromarray(rgb).convert("RGB")
            rgb_frame = rgb_frame.resize((320,240))
            rgb_frame.save(os.path.join(curren_image_path, f"frame_{step_id}.jpg"))
            action = reference_actions.pop(0) 
            observation = env.step(action)  
            step_id += 1
        output_dict["time_per_episode"] = time.time() - episode_start_time
        output_dict["episode_id"] = int(episode.episode_id)
        result_queue.put(output_dict)
    env.close()

def process_single_dataset(dataset_name, save_image, split_num):
    CONFIG_PATH = CONFIG[dataset_name]['config_path']
    ANNOT_PATH = CONFIG[dataset_name]['annotation_path']
    IMAGE_PATH = CONFIG[dataset_name]['image_path']
    annotations = []
    with open(ANNOT_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            annotations.append(item)
    annotations = sorted(annotations, key=lambda x: x["episode_id"])
    falici = {annotations[i]['episode_id']:i for i in range(len(annotations))}
    
    num_episodes = len(annotations)
    env_config = get_config(CONFIG_PATH)
    dataset = habitat.datasets.make_dataset(id_dataset=env_config.habitat.dataset.type, config=env_config.habitat.dataset)
    dataset_splits = dataset.get_splits(split_num, allow_uneven_splits=True)
    sub_annotations = []
    for sub_dataset in dataset_splits:
        tmp = []
        for item in sub_dataset.episodes:
            tmp.append(copy.deepcopy(annotations[falici[int(item.episode_id)]]))
        sub_annotations.append(tmp)
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(split_num):
        worker_args = (
                result_queue, env_config,
                sub_annotations[i], dataset_splits[i], 
                save_image,
                IMAGE_PATH
            )
        p = mp.Process(target=extract_data, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="generating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            episode_id = result['episode_id']
            pbar.update(1)
            pbar.set_postfix(time_per_episode = result['time_per_episode'])
    
    for p in processes:
        p.join()

        
    

def main(dataset2process, save_image, num_thread):
    for dataset_name in dataset2process:
        process_single_dataset(dataset_name, save_image, num_thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    parser.add_argument("--save_image", action="store_true", default=False)
    parser.add_argument("--num_thread", type=int, default=16)
    
    args = parser.parse_args()
    print(args.dataset_name)
    main(args.dataset_name, args.save_image, args.num_thread)