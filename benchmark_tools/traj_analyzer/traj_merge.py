from posixpath import dirname


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge multiple trajectory files into a single trajectory file."
    )
    parser.add_argument(
        "--split_traj_path",
        default='/root/NavDataGeneration/data/traj_datasets/objectnav/cloudrobo_v1_l3mvn_all/train/content_split',
        help="Path to the directory containing split trajectory files.",
    )
    parser.add_argument(
        "--merged_traj_path",
        default='/root/content',
        help="Path to save the merged trajectory file.",
    )
    parser.add_argument(
        "--task_path",
        default='data/task_datasets/objectnav/cloudrobo_v1/train/content',
        help="task datasets path, conculde scene names from here.",
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
    ok_scene_names = []
    for _, _, filenames in os.walk(args.task_path):
        for filename in filenames:
            if filename.endswith('.json.gz'):
                scene_name = filename.replace('.json.gz', '')
                scene_names.append(scene_name)
    for _, _, filenames in os.walk(args.merged_traj_path):
        for filename in filenames:
            if filename.endswith('.json.gz'):
                scene_name = filename.replace('.json.gz', '')
                if scene_name not in ok_scene_names:
                    ok_scene_names.append(scene_name)
    scene_names = [name for name in scene_names if name not in ok_scene_names]
    print(f'scene_names: {scene_names}')
    print(f'Number of scenes: {len(scene_names)}')

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
            print(f"Processing scene {scene_name} in dir {split_dir_name}...")
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
        json_str = json.dumps(merged_traj_data)
        with gzip.open(merged_filepath, 'wt', encoding='utf-8') as f:
            f.write(json_str)

        
        return scene_episode_num
    
    #with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    #    results = list(executor.map(process_scene, scene_names))
    #    total_episodes = sum(results)
    for scene_name in scene_names:
        total_episodes += process_scene(scene_name)

    print(f"Total episodes merged: {total_episodes}")


if __name__ == "__main__":
    main()