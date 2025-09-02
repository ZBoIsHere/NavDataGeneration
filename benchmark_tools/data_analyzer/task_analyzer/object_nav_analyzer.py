import os
import gzip
import json


def load_filenames(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def process_json_files_from_list(txt_path, search_directory):

    filenames = load_filenames(txt_path)
    for filename in filenames:

        file_path = os.path.join(search_directory, filename + '.json.gz')
        if not file_path:
            print(f"未找到文件: {filename}")
            continue

        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                json_data = json.load(f)
                episodes = json_data.get('episodes', [])
                print(f"{filename}: {len(episodes)}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")


txt_path = '/app/data/z00562901/NavDataGeneration/NavTrajSampleGeneration/L3MVN/hm3d_sem_v1_train_scenes.txt'
search_directory = '/app/data/z00562901/NavDataGeneration/data/objectgoal_hm3d/train/content'
process_json_files_from_list(txt_path, search_directory)
