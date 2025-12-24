## PYTHON 
## ******************************************************************** ##
## author: CTO_TI_FBSYJG
## create time: 2025/09/23 12:03:01 GMT+08:00
## ******************************************************************** ##
import os
import argparse
import uuid
import gzip
import json

os.chdir('/mnt/sfs-turbo-workflow/data-platform/')

# UUID 生成逻辑
def generate_uuid(scene_type, scene_name, split, start_position, start_rotation, object_category):
    namespace = uuid.NAMESPACE_DNS
    key = f"{scene_type}:{scene_name}:{split}:{start_position}:{start_rotation}:{object_category}"
    return str(uuid.uuid5(namespace, key))

def main(args):
    if not os.path.exists(args.input_path):
        print(f"❌ 输入路径不存在: {args.input_path}")
        # 退出码 1 表示错误
        os._exit(1)
        
    # output是一个二维数组，其中第一维是场景名称，第二维是uuid以“，”分隔的字符串
    output = []
    for filename in os.listdir(args.input_path):
        # 如果filename以.json.gz结尾 并且 filename以数字开头， 则处理该文件
        if filename.endswith(".json.gz") and filename[0].isdigit():
            filepath = os.path.join(args.input_path, filename)
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                episodes = data.get("episodes", [])
                # 将前3个episode进行处理
                episodes = episodes[:3]

                scene_name = filename.replace(".json.gz", "")   
                uuids = ''

                for ep in episodes:
                    start_position = ep["start_position"]
                    start_rotation = ep["start_rotation"]
                    object_category = ep["object_category"]

                    record_id = generate_uuid(
                        args.scene_type, scene_name, args.split,
                        start_position, start_rotation, object_category
                    )
                    uuids += record_id + ","
                output.append([scene_name, uuids])
    
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="列举 ObjectNav 数据集中的场景名称")
    parser.add_argument("--input_path", default="data/task_datasets/objectnav/hm3d_v1/train/content", help="数据集路径")
    parser.add_argument("--scene_type", default="hm3d_v1", help="场景类型")
    parser.add_argument("--split", default="train", help="数据集划分类型")

    args = parser.parse_args()
    main(args)