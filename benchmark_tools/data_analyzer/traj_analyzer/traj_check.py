from ast import arg
import os
import gzip
import json

def check_traj_format(args):
    len_bet_500_1000 = 0
    len_large_1000 = 0
    # 遍历文件夹下所有的json.gz文件，打印每个json_data['episodes']中episode的reference_replay的第一个action和最后一个action
    for _, _, filenames in os.walk(args.traj_path):
        for filename in filenames:
            if filename.endswith('.json.gz'):
                # 如果不以ACZZi开头，就跳过
                if not filename.startswith('ACZZi'):
                    continue
                filepath = os.path.join(args.traj_path, filename)
                print(f"Processing file: {filename}")
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        json_data = json.load(f)
                        episodes = json_data.get('episodes', [])
                        for i in range(len(episodes)):
                            if i != 200:
                                continue
                            episode = episodes[i]
                            replay = episode.get('reference_replay', [])
                            if replay:
                                first_action = replay[0].get('action')
                                last_action = replay[-1].get('action')
                                # 如果首尾的action不是'STOP'，打印警告
                                if first_action != 'STOP':
                                    print(f"Warning: {filename} - Episode {episode.get('episode_id')} first action is not STOP, it is {first_action}")
                                if last_action != 'STOP':
                                    print(f"Warning: {filename} - Episode {episode.get('episode_id')} last action is not STOP, it is {last_action}")
                                # 遍历所有的replay，如果action不在['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']，打印警告
                                angle = 0
                                for step in replay[1:-1]:  # 排除首尾的STOP
                                    action = step.get('action')
                                    if action not in ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']:
                                        print(f"Warning: {filename} - Episode {episode.get('episode_id')} has invalid action {action}")
                                    if action in ['LOOK_UP', 'LOOK_DOWN']:
                                        if action == 'LOOK_UP':
                                            angle += 1
                                        else:
                                            angle -= 1
                                        #print(f"Warning: {filename} - Episode {episode.get('episode_id')} has LOOK_UP or LOOK_DOWN action.")

                                if angle != 0:
                                    print(f"Warning: {filename} - Episode {episode.get('episode_id')} has unbalanced LOOK_UP/LOOK_DOWN actions, net angle change: {angle}")
                                # 如果replay的长度大于500，打印警告
                                if len(replay) > 500 and len(replay) <= 1000:
                                    len_bet_500_1000 += 1
                                elif len(replay) > 1000:
                                    len_large_1000 += 1
                            else:
                                print(f"{filename} - Episode {episode.get('episode_id')}: No reference_replay found.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    print(f"Number of episodes with reference_replay length between 500 and 1000: {len_bet_500_1000}")
    print(f"Number of episodes with reference_replay length greater than 1000: {len_large_1000}")

def calculate_episode_sum(episode_path):
    total_episodes = 0
    for _, _, filenames in os.walk(episode_path):
        for filename in filenames:
            if filename.endswith('.json.gz'):
                filepath = os.path.join(episode_path, filename)
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        json_data = json.load(f)
                        episodes = json_data.get('episodes', [])
                        print(f"{filename}: {len(episodes)} episodes")
                        total_episodes += len(episodes)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    return total_episodes

def check_episode_range(episode_path, episode_min, episode_max):
    for _, _, filenames in os.walk(episode_path):
        for filename in filenames:
            if filename.endswith('.json.gz'):
                filepath = os.path.join(episode_path, filename)
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        json_data = json.load(f)

                        # 检查 'episodes' 是否存在且不为空
                        episodes = json_data.get('episodes', [])
                        if episodes:
                            first_id = int(episodes[0].get('episode_id'))
                            last_id = int(episodes[-1].get('episode_id'))
                            if first_id < episode_min or last_id > episode_max:
                                print(f"Warning: {filename} has episode IDs out of expected range ({episode_min}-{episode_max})")
                            print(f"{filename}: Episode ID range is {first_id} to {last_id}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


def query_missing_scene(episode_path, task_dataset_list):
    for _, _, filenames in os.walk(episode_path):
        if len(filenames) != len(task_dataset_list)/2:
            #print(f"Warning: Expected 160 files, found {len(filenames)} files in {episode_dir}\n")
            filenames_list = [filename.replace('.json.gz', '') for filename in filenames if filename.endswith('.json.gz')]
            # find the missing index from filenames_list to task_dataset_list
            missing_filenames_index = [i for i, name in enumerate(task_dataset_list) if name not in filenames_list]
            print(f"Missing files index: {missing_filenames_index}")

def get_scene_list(scene_list_file):
    scene_names = []
    with open(scene_list_file, 'r') as f:
        for line in f:
            scene_name = line.strip()
            scene_names.append(scene_name)
    return scene_names

# 判断episode_path下面的所有json.gz文件中，episode_id是不是递增的，有没有重复的
def check_episode_id_increasing(episode_path):
    for _, _, filenames in os.walk(episode_path):
        for filename in filenames:
            print(f"Checking {filename} for episode_id consistency...")
            episode_ids = []
            if filename.endswith('.json.gz'):
                filepath = os.path.join(episode_path, filename)
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        json_data = json.load(f)
                        episodes = json_data.get('episodes', [])
                        for episode in episodes:
                            episode_id = int(episode.get('episode_id'))
                            if episode_id in episode_ids:
                                print(f"Warning: Duplicate episode_id {episode_id} found in {filename}")
                            episode_ids.append(episode_id)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            # 判断episode_ids是不是递增的， 不要求是连续递增的
            if episode_ids != sorted(episode_ids):
                print(f"Warning: episode_ids in {filename} are not in increasing order.")
            else:
                print(f"All episode_ids in {filename} are in increasing order.")

import argparse

def main():
    argparser = argparse.ArgumentParser(description="Check trajectory files in a specified directory.")

    argparser.add_argument(
        "--task_path",
        type=str,
        default='/app/data/z00562901/NavDataGeneration/data/objectgoal_hm3d/train/content',
        help="task path"
    )

    # traj_path
    argparser.add_argument(
        "--traj_path",
        type=str,
        default='/root/NavDataGeneration/data/traj_datasets/objectnav/hm3d_v1_hd/train/content',
        help="Path to the trajectory dataset directory."
    )



    args = argparser.parse_args()

    #scene_list = get_scene_list(args.scene_list_file)
    #print(f"Number of scenes: {len(scene_list)}")
    #print(f"Scene list: {scene_list}")

    #task_episode_num = calculate_episode_sum(args.task_path)
    #print(f"Total episodes in task path: {task_episode_num}")

    #traj_episode_num = calculate_episode_sum(args.traj_path)
    #print(f"Total episodes in traj path: {traj_episode_num}")
    #check_episode_id_increasing(args.traj_path)
    check_traj_format(args)
    #check_traj_quanlity(args)



if __name__ == "__main__":
    main()