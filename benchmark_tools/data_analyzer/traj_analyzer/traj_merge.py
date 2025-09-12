

from posixpath import dirname

import scipy as sp


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge multiple trajectory files into a single trajectory file."
    )
    parser.add_argument(
        "--split_traj_path",
        default='/app/data/z00562901/NavDataGeneration/data/traj_datasets/objectnav/hm3d_v1_hd',
        help="Path to the directory containing split trajectory files.",
    )
    parser.add_argument(
        "--merged_traj_path",
        default='/app/data/z00562901/NavDataGeneration/data/traj_datasets/objectnav/hm3d_v1_hd/train/content/',
        help="Path to save the merged trajectory file.",
    )
    # split length
    parser.add_argument(
        "--split_length",
        type=int,
        default=100,
        help="Number of trajectories per split file.",
    )
    # start_episode_id
    parser.add_argument(
        "--start_episode_id",
        type=int,
        default=0,
        help="Starting episode ID for naming the merged file.",
    )
    # end_episode_id
    parser.add_argument(
        "--end_episode_id",
        type=int,
        default=1199,
        help="Ending episode ID for naming the merged file.",
    )


    args = parser.parse_args()
    # os.walk args.split_traj_path and sort, to get episode range like 0-99, 100-199, ...
    import os
    split_dir_names = []
    for _, split_dir_names, _ in os.walk(args.split_traj_path):
        # 除去不是episode_num开头的目录
        split_dir_names = [d for d in split_dir_names if d.startswith('episode_num_')]
        # get dir_names and sort, get like episode_num_0-99, episode_num_100-199, ...
        split_dir_names.sort(key=lambda x: int(x.split('_')[-1].split('-')[0]))
        break

    print(f'split_dir_names: {split_dir_names}')

    # 所有的场景名称在一个.txt文件中，每行一个场景名称
    scene_names = []
    with open('/app/data/z00562901/NavDataGeneration/NavTrajSampleGeneration/L3MVN/hm3d_sem_v1_train_scenes.txt', 'r') as f:
        for line in f:
            scene_name = line.strip()
            scene_names.append(scene_name)
    print(f'scene_names: {scene_names}')
    print(f'Number of scenes: {len(scene_names)}')

    # 依次遍历scene_names, 然后把split_dir_name中场景名称相同的*.json.gz文件合并，其中合并的内容是json_data['episodes']的列表，其他json_data的字段，每个场景都一样，使用第一个文件的即可
    # 如果某个场景的文件在某个split_dir_names中不存在，则跳过该文件
    # 要求逐个对每个场景进行合并，而不是等到所有文件都读完再进行合并

    import gzip
    import json
    #merged_traj_data = {}
    total_episodes = 0
    # 并行化处理每个scene_name
    import concurrent.futures
    def process_scene(scene_name):
        merged_traj_data = {
            'episodes': [],
            'category_to_task_category_id': {},
            'category_to_scene_annotation_category_id': {},  
            'goals_by_category': {},    
        }
        for split_dir_name in split_dir_names:
            split_dir_path = os.path.join(args.split_traj_path, split_dir_name)
            filename = f"{scene_name}.json.gz"
            filepath = os.path.join(split_dir_path, filename)
            if os.path.exists(filepath):
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if not merged_traj_data['category_to_task_category_id']:
                            merged_traj_data['category_to_task_category_id'] = json_data.get('category_to_task_category_id', {})
                        if not merged_traj_data['category_to_scene_annotation_category_id']:
                            merged_traj_data['category_to_scene_annotation_category_id'] = json_data.get('category_to_scene_annotation_category_id', {})  
                        if not merged_traj_data['goals_by_category']:
                            merged_traj_data['goals_by_category'] = json_data.get('goals_by_category', {})    
                        merged_traj_data['episodes'].extend(json_data.get('episodes', []))
                        #print(f"Merged {len(json_data.get('episodes', []))} episodes from {filename} from dir {split_dir_name}.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"File {filename} does not exist in {split_dir_path}, skipping.")

        # save merged_traj_data to args.merged_traj_path, filename is like '{scene_name}.json.gz'
        merged_filepath = os.path.join(args.merged_traj_path, f"{scene_name}.json.gz")
        os.makedirs(dirname(merged_filepath), exist_ok=True)
        scene_episode_num = len(merged_traj_data['episodes'])
        print(f"Scene {scene_name} has {scene_episode_num} episodes after merging.")
        with gzip.open(merged_filepath, 'wt', encoding='utf-8') as f:
            json.dump(merged_traj_data, f)
        
        return scene_episode_num
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_scene, scene_names))
        total_episodes = sum(results)

    print(f"Total episodes merged: {total_episodes}")




if __name__ == "__main__":
    main()