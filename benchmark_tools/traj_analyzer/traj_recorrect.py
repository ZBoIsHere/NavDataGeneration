'''
轨迹的格式如下

{
    "episodes": [
        {
            "episode_id": "0",
            "metrics": {
                "success": 1,
            },
    ]
}
'''

# 写一段程序，解压json.gz文件，读取其中的轨迹数据，把metrics删除，然后重新保存为json.gz文件
import json
import gzip
import os
from typing import Dict, Any
from tqdm import tqdm
def recorrect_trajectory_file(input_file: str, output_file: str) -> None:
    """
    解压json.gz文件，读取其中的轨迹数据，把metrics删除，然后重新保存为json.gz文件

    Args:
        input_file (str): 输入的json.gz文件路径
        output_file (str): 输出的json.gz文件路径
    """
    # 读取压缩文件
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # 删除每个episode中的metrics字段
    for episode in tqdm(data.get("episodes", []), desc="Processing episodes"):
        if "metrics" in episode:
            del episode["metrics"]

    # 保存修改后的数据到新的压缩文件
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(data, f)

if __name__ == "__main__":
    input_path = "/root/NavDataGeneration/data/traj_datasets/objectnav/cloudrobo_v1_l3mvn_all/train/content/shanghai-lianqiuhu-b11-4f-big-2025-07-15_11-40-37.json.gz"
    output_path = "/root/NavDataGeneration/data/traj_datasets/objectnav/cloudrobo_v1_l3mvn_all/train/content/shanghai-lianqiuhu-b11-4f-big-2025-07-15_11-40-37.json.gz"
    recorrect_trajectory_file(input_path, output_path)