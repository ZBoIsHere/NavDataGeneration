## PYTHON 
## ******************************************************************** ##
## author: CTO_TI_FBSYJG
## create time: 2025/09/23 12:03:01 GMT+08:00
## ******************************************************************** ##
import os
import argparse

os.chdir('/mnt/sfs-turbo-workflow/data-platform/')

def main(args):
    if not os.path.exists(args.input_path):
        print(f"❌ 输入路径不存在: {args.input_path}")
        # 退出码 1 表示错误
        os._exit(1)
        
    output = []
    for filename in os.listdir(args.input_path):
        if filename.endswith(".json.gz"):
            scene_name = filename.replace(".json.gz", "")
            output.append([scene_name])
    
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="列举 ObjectNav 数据集中的场景名称")
    parser.add_argument("--input_path", default="data/task_datasets/objectnav/hm3d_v1/train/content", help="数据集路径")

    args = parser.parse_args()
    main(args)
