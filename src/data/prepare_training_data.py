import json
import os
import random
import re
import copy
from tqdm import tqdm
import argparse


forward_distance = 25
turn_angle = 15

def noise_action(action_id):
    p = random.random()
    if p < 0.7 and action_id != 0:
        return random.randint(1,3)
    else:
        return action_id
def action_id_to_str(action_id):
    # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
    if action_id == 0:
        return "stop"
    elif action_id == 1:
        return f"forward {forward_distance} cm"
    elif action_id == 2:
        return f"turn left {turn_angle} degree"
    elif action_id == 3:
        return f"turn right {turn_angle} degree"
    else:
        raise ValueError(f"Invalid action ID: {action_id}")

def combine(action1,action2):
    idx = action1.rfind(', ')
    subaction0 = action1[:idx+1]+' ' if action1[:idx+1] != '' else action1[:idx+1]
    subaction1 = action1[idx+1:]
    match1 = re.search(r'-?\d+', subaction1)
    match1 = int(match1.group())
    match2 = re.search(r'-?\d+', action2)
    match2 = int(match2.group())
    if "forward" in subaction1:
        if match1+match2 <= 3*forward_distance:
            return f"{subaction0}forward {match1+match2} cm"
        else:
            return None
    elif "turn left" in subaction1:
        if match1+match2 <= 3*turn_angle:
            return f"{subaction0}turn left {match1+match2} degree"
        else:
            return None
    elif "turn right" in subaction1:
        if match1+match2 <= 3*turn_angle:
            return f"{subaction0}turn right {match1+match2} degree"
        else:
            return None
    else:
        raise ValueError(f"Invalid action: {action1}")



def spilt_method(x):
    return x.split("_")[1].split(".")[0]



config = {
    'r2r': {
        'image_path': 'data/images/r2r',
        'annotation_path': 'data/sub_dataset/r2r.jsonl',
        'split_method': spilt_method,
    },
    'rxr': {
        'image_path': 'data/images/rxr',
        'annotation_path': 'lam_vln/data/sub_dataset/rxr.jsonl',
        'split_method': spilt_method,
    },
    'envdrop': {
        'image_path': 'data/images/envdrop',
        'annotation_path': 'lam_vln/data/sub_dataset/envdrop.jsonl',
        'split_method': spilt_method,
    },
    'scalevln': {
        'image_path': 'data/images/scalevln',
        'annotation_path': 'lam_vln/data/sub_dataset/scalevln.jsonl',
        'split_method': spilt_method,
    },
}


def process_single_type(selected_subset_list, system_prompt, prompt_template, task_type):
    data2save = []
    for subset in tqdm(selected_subset_list):

        subset_config = config[subset]
        image_path = subset_config['image_path']
        annotation_path = subset_config['annotation_path']
        split_method = subset_config['split_method']
        sub_image_path = config.get('sub_image_path', None)

        # read annotations
        annotation = []
        with open(annotation_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                annotation.append(item)
            
        for episode_item in annotation:
            episode_id = episode_item['episode_id']
            video_id = episode_item['video_id']
            instruction = episode_item['instruction']
            actions = episode_item['actions']
            assert actions[-1] == 0

            episode_image_path = os.path.join(image_path, str(video_id))
            if sub_image_path is not None:
                episode_image_path = os.path.join(episode_image_path, sub_image_path)
            episode_image_list = os.listdir(episode_image_path)
            episode_image_list = sorted(episode_image_list, key=lambda x: int(split_method(x)))
            episode_image_list = [episode_image_path+'/'+image for image in episode_image_list]
            if len(episode_image_list) != len(actions) + 1:
                episode_image_list.append(episode_image_list[-1])
            assert len(episode_image_list) == len(actions) + 1

            tmp_data = {
                "system": system_prompt,
                "conversations": [],
                "action_history":[],
                "episode_id":str(episode_id),
                # 'idm_input': [],
                "task type": task_type,
            }
            if task_type == "vln":
                formated_instruction = prompt_template.format(instruction)
                tmp_data["conversations"].append({"from": "user", "value": formated_instruction,"image":[episode_image_list[0]]})
                tmp_data["conversations"].append({"from": "assistant", "value": action_id_to_str(actions[0])}) 
                last_action = actions[0]
                pending_rgb_list = []
                for i in range(1, len(actions)):
                    prob = random.random()
                    if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action)) is not None:
                        tmp_data["conversations"][-1]['value'] = combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action))
                        pending_rgb_list.append(episode_image_list[i])
                    else:
                        count = tmp_data["conversations"][-1]['value'].count(',')
                        if count < 2:
                            tmp_data["conversations"][-1]['value'] += ', ' + action_id_to_str(actions[i])
                            pending_rgb_list.append(episode_image_list[i])
                        else:
                            # tmp_data['idm_input'] = [tmp_data["conversations"][0]['image'][-1], episode_image_list[i]]
                            data2save.append(copy.deepcopy(tmp_data))
                            tmp_data["action_history"].append(tmp_data["conversations"][1]['value'])
                            while len(pending_rgb_list)!= 0:
                                item = pending_rgb_list.pop(0)
                                tmp_data["conversations"][0]['image'].append(item)
                            tmp_data["conversations"][0]['image'].append(episode_image_list[i])
                            # while len(tmp_data["conversations"][0]['image']) > max_images:
                            #     tmp_data["conversations"][0]['image'] = tmp_data["conversations"][0]['image'][1:]
                            tmp_data["conversations"][1]['value'] = action_id_to_str(actions[i])
                            # assert len(conversation["action_history"]) == len(conversation["conversations"][0]['image']) - 1
                    last_action = actions[i]
                # tmp_data['idm_input'] = [tmp_data["conversations"][0]['image'][-1], episode_image_list[i+1]]
                data2save.append(copy.deepcopy(tmp_data))
            elif task_type == "idm":
                formated_instruction = prompt_template
                tmp_data["conversations"].append({"from": "user", "value": formated_instruction,"image":[episode_image_list[0], episode_image_list[1]]})
                tmp_data["conversations"].append({"from": "assistant", "value": action_id_to_str(actions[0])}) 
                last_action = actions[0]
                for i in range(1, len(actions)):
                    prob = random.random()
                    if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action)) is not None:
                        tmp_data["conversations"][-1]['value'] = combine(tmp_data["conversations"][-1]['value'],action_id_to_str(last_action))
                        tmp_data["conversations"][-2]['image'][-1] = episode_image_list[i+1]
                    else:
                        count = tmp_data["conversations"][-1]['value'].count(',')
                        if count < 2:
                            tmp_data["conversations"][-1]['value'] += ', ' + action_id_to_str(actions[i])
                            tmp_data["conversations"][-2]['image'][-1] = episode_image_list[i+1]
                        else:
                            data2save.append(copy.deepcopy(tmp_data))
                            tmp_data["action_history"].append(tmp_data["conversations"][1]['value'])
                            tmp_data["conversations"][0]['image'] = [episode_image_list[i],episode_image_list[i+1]]
                            tmp_data["conversations"][1]['value'] = action_id_to_str(actions[i])
                            # assert len(conversation["action_history"]) == len(conversation["conversations"][0]['image']) - 1
                    last_action = actions[i]
                if tmp_data["conversations"][1]['value'] != 'stop':
                    data2save.append(copy.deepcopy(tmp_data))

    

def main(selected_subset_list, task_type_list, output_path):
    system_prompt = "You are a helpful assistant."
    vln_prompt_template = "Imagine you are a robot programmed for navigation tasks. "\
        "You have been given a video of historical observations and an image of the current observation. "\
        "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
        "which could involve turning left or right by a specific degree or moving forward a certain distance."

    idm_prompt_template = "Imagine you are a robot programmed for navigation tasks. "\
        "You have been given an image of current view and an image of the goal view. "\
        "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "\
        "which could involve turning left "\
        "or right by a specific degree or moving forward a certain distance."
    
    data2save = []

    for task_type in task_type_list:
        if task_type == "vln":
            prompt_template = vln_prompt_template
        elif task_type == "idm":
            prompt_template = idm_prompt_template

        data2save.extend(process_single_type(selected_subset_list, system_prompt, prompt_template, task_type))

    print(f"total number of samples = {len(data2save)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data2save:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(len(data2save))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    parser.add_argument(
        "--task_type",
        nargs="+",
        default=["vln", "idm"],
    )
    parser.add_argument("--output_path", type=str, default='data/navida_train_data.jsonl')
    args = parser.parse_args()


    main(args.dataset_name, args.task_type, args.output_path)