import json
import gzip
from pathlib import Path
import argparse

def modify_trajectory_file(args, traj_file):
    relative_path = traj_file.relative_to(args.input_dir).parent
    output_subdir = args.output_dir / relative_path
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file_name = traj_file.name
    output_file_path = output_subdir / output_file_name
    input_file = str(traj_file)
    output_file = str(output_file_path)
    # 如果输出文件已存在，则跳过处理
    if Path(output_file).exists():
        print(f"Output file already exists, skipping: {output_file}")
        return


    print(f"Processing file: {input_file}")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # 要求data里面除了episodes字段外，其他字段保持不变
    modified = False
    for episode in data.get('episodes', []):
        if episode.get('reference_replay'):
            first_action = episode['reference_replay'][0].get('action')
            if first_action != 'STOP':
                episode['reference_replay'].insert(0, {'action': 'STOP'})
                modified = True
            else:
                print(f"'STOP' action already present in first position for episode id: {episode.get('episode_id')}")

    if modified:
        print(f"Modified file: {input_file}")
        json_data = json.dumps(data, indent=None)
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            f.write(json_data)
        print(f"Saved modified file to: {output_file}")


def batch_modify_trajectories(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_files = list(input_path.rglob('*.json.gz'))
    from multiprocessing import Pool
    with Pool(processes=2) as pool:
        pool.starmap(modify_trajectory_file, [(args, traj_file) for traj_file in traj_files])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch modify trajectory files.")
    parser.add_argument('--input_dir', type=str, default='/root/content', help="Directory containing trajectory files.")
    parser.add_argument('--output_dir', type=str, default='/root/content2' , help="Directory to save modified trajectory files.")
    args = parser.parse_args()

    batch_modify_trajectories(args)