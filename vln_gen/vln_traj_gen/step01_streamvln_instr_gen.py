import os
import sys
import gzip
import json
import shutil
import argparse
from glob import glob
from tqdm import tqdm
import uuid

def generate_vln_uuid(scene_type, scene_name, split, start_position, start_rotation, goal_position):
    """生成唯一的episode ID"""
    namespace = uuid.NAMESPACE_DNS
    key = f"{scene_type}:{scene_name}:{split}:{start_position}:{start_rotation}:{goal_position}"
    return str(uuid.uuid5(namespace, key))[:8]

def validate_episode_range(episode_range):
    """验证episode_range格式是否正确"""
    if not episode_range.count('-') == 1:
        raise ValueError(f"episode_range格式无效，应为'start-end'，当前值: {episode_range}")
    
    try:
        start_ep, end_ep = map(int, episode_range.split('-'))
    except ValueError:
        raise ValueError(f"episode_range应为整数范围，当前值: {episode_range}")
    
    if start_ep % 100 != 0 or end_ep % 100 != 99:
        raise ValueError(f"episode_range必须以100为单位划分 (例如 3100-3199)，当前值: {episode_range}")
    
    return start_ep, end_ep

def get_episode_ranges(start_ep, end_ep):
    """生成episode范围列表"""
    episode_range_list = []
    for ep_start in range(start_ep, end_ep + 1, 100):
        ep_end = min(ep_start + 99, end_ep)
        episode_range_list.append(f"episode_num_{ep_start}-{ep_end}")
    return episode_range_list

def load_json_gz(file_path):
    """安全加载gzip压缩的json文件"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {str(e)}")
        return None

def delete_old_files(annotation_file, videos_path):
    """删除旧的非压缩文件和videos目录"""
    # 删除非压缩annotation文件
    non_gz_file = annotation_file.replace('.gz', '')
    if os.path.exists(non_gz_file):
        try:
            os.remove(non_gz_file)
            print(f"已删除旧的annotation文件: {non_gz_file}")
        except Exception as e:
            print(f"警告: 无法删除旧文件 {non_gz_file}: {str(e)}")
    
    # 删除videos目录
    if os.path.exists(videos_path):
        try:
            shutil.rmtree(videos_path)
            print(f"已删除videos目录: {videos_path}")
        except Exception as e:
            print(f"警告: 无法删除videos目录 {videos_path}: {str(e)}")

def validate_instructions(instructions_data, scene_name):
    """验证instruction数据有效性"""
    if not instructions_data or 'episodes' not in instructions_data:
        print(f"错误: instruction文件中缺少有效的'episodes'字段 - {scene_name}")
        return False, 0
    
    episodes = instructions_data['episodes']
    if not episodes:
        print(f"错误: instruction文件中没有episodes数据 - {scene_name}")
        return False, 0
    
    # 检查是否所有instruction都为空
    valid_count = 0
    for ep in episodes:
        if 'instruction' in ep and ep['instruction'] and len(ep['instruction']) > 0:
            valid_count += 1
    
    if valid_count == 0:
        print(f"警告: 所有episodes的instruction都为空 - {scene_name}")
        return False, 0
    
    return True, valid_count

def check_consistency(episodes, instr_episodes, scene_name, image_base_path):
    """检查动作数量和图片数量的一致性"""
    valid_episodes = []
    inconsistencies = 0
    
    for episode in episodes:
        # 生成episode ID
        episode_id = generate_vln_uuid(
            'hm3d_v1',
            scene_name,
            'train',
            episode["start_position"],
            episode["start_rotation"],
            episode["goals"][0]["position"]
        )
        
        # 获取动作数量
        instr_episode = instr_episodes.get(episode_id, {})
        actions = instr_episode.get("actions", [])
        action_count = len(actions)
        
        # 获取图片数量
        image_dir = os.path.join(image_base_path, scene_name, episode_id)
        image_count = len(os.listdir(image_dir)) if os.path.exists(image_dir) else 0
        
        # 检查一致性
        if action_count != image_count:
            inconsistencies += 1
            print(f"  不一致: Episode={episode_id}, 动作数量={action_count}, 图片数量={image_count}")
        else:
            valid_episodes.append((episode, instr_episode))
    
    if inconsistencies > 0:
        print(f"  场景 {scene_name} 有 {inconsistencies} 个不一致的episodes")
    
    return valid_episodes

def process_scene(args, scene_name, vln_path, stream_path):
    """处理单个场景：整合instruction插入和一致性检查"""
    # 构建文件路径
    instruction_file = os.path.join(vln_path, "instructions", f"{scene_name}.json.gz")
    annotation_file = os.path.join(stream_path, "annotations", f"{scene_name}.json.gz")
    videos_path = os.path.join(stream_path, "videos")
    image_base_path = os.path.join(stream_path, "images")
    
    # 验证必要文件
    if not os.path.exists(instruction_file):
        print(f"跳过: instruction文件不存在 - {instruction_file}")
        return 0
    if not os.path.exists(annotation_file):
        print(f"跳过: annotation文件不存在 - {annotation_file}")
        return 0
    
    # 加载数据
    instructions_data = load_json_gz(instruction_file)
    annotations_data = load_json_gz(annotation_file)
    
    # 验证instruction数据
    is_valid, valid_count = validate_instructions(instructions_data, scene_name)
    if not is_valid:
        return 0
    
    # 构建instruction字典
    instruction_dict = {}
    instr_episodes = {}
    for episode in instructions_data['episodes']:
        if 'episode_id' not in episode:
            continue
        ep_id = episode['episode_id']
        if 'instruction' in episode and episode['instruction'] and len(episode['instruction']) > 0:
            instruction_dict[ep_id] = episode['instruction']
        if 'actions' in episode:
            instr_episodes[ep_id] = episode
    
    # 执行一致性检查
    valid_episodes = check_consistency(
        annotations_data, 
        instr_episodes, 
        scene_name, 
        image_base_path
    )
    
    if not valid_episodes:
        print(f"警告: 场景 {scene_name} 没有通过一致性检查的有效episodes")
        return 0
    
    # 过滤并更新annotations
    filtered_annotations = []
    for annotation, instr_ep in valid_episodes:
        ep_id = annotation['id']
        if ep_id in instruction_dict:
            annotation['instructions'] = instruction_dict[ep_id]
            filtered_annotations.append(annotation)
    
    if not filtered_annotations:
        print(f"警告: 场景 {scene_name} 没有过滤后的有效annotations")
        return 0
    
    # 删除旧文件
    delete_old_files(annotation_file, videos_path)
    
    # 保存新annotations
    save_annotations(filtered_annotations, annotation_file)
    return len(filtered_annotations)

def save_annotations(annotations_data, annotation_file):
    """保存annotations到json和json.gz格式"""
    json_str = json.dumps(annotations_data, indent=4)
    non_gz_file = annotation_file.replace('.gz', '')
    
    # 保存非压缩文件
    try:
        with open(non_gz_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  已保存非压缩annotation: {non_gz_file}")
    except Exception as e:
        print(f"错误: 无法保存非压缩文件 {non_gz_file}: {str(e)}")
        return False
    
    # 保存压缩文件
    try:
        with gzip.open(annotation_file, 'wt', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  已保存压缩annotation: {annotation_file}")
        return True
    except Exception as e:
        print(f"错误: 无法保存压缩文件 {annotation_file}: {str(e)}")
        # 清理非压缩文件
        if os.path.exists(non_gz_file):
            os.remove(non_gz_file)
        return False

def process_all_scenes(args):
    """处理所有episode_range下的所有场景文件"""
    try:
        start_ep, end_ep = validate_episode_range(args.episode_range)
    except ValueError as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    episode_range_list = get_episode_ranges(start_ep, end_ep)
    print(f'处理的episode范围: {episode_range_list}')
    
    total_processed = 0
    for episode_range in episode_range_list:
        # 构建当前范围的路径
        current_vln_path = os.path.join(os.path.dirname(args.vln_topo_path), episode_range)
        current_stream_path = os.path.join(os.path.dirname(args.streamvln_data_path), episode_range)
        
        print(f'\n处理范围: {episode_range}')
        print(f'VLN路径: {current_vln_path}')
        print(f'Stream路径: {current_stream_path}')
        
        # 验证路径存在性
        if not os.path.exists(current_vln_path):
            print(f"跳过: VLN路径不存在 - {current_vln_path}")
            continue
        if not os.path.exists(current_stream_path):
            print(f"跳过: Stream路径不存在 - {current_stream_path}")
            continue
        
        # 处理每个场景文件
        scene_files = [f for f in os.listdir(current_vln_path) if f.endswith(".json.gz")]
        if not scene_files:
            print(f"警告: 在 {current_vln_path} 中没有找到场景文件")
            continue
        
        print(f"  发现 {len(scene_files)} 个场景文件")
        for scene_file in tqdm(scene_files, desc=f"处理 {episode_range}"):
            scene_name = scene_file.replace(".json.gz", "")
            processed = process_scene(args, scene_name, current_vln_path, current_stream_path)
            total_processed += processed
    
    print(f"\n总计: 成功处理 {total_processed} 个有效episodes")

def main():
    parser = argparse.ArgumentParser(description='VLN数据处理工具')
    parser.add_argument('--vln_topo_path', type=str, 
                        default='data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199',
                        help='VLN拓扑数据路径')
    parser.add_argument('--streamvln_data_path', type=str,
                        default='data/traj_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199',
                        help='StreamVLN数据路径')
    parser.add_argument('--episode_range', type=str, default='3100-9999',
                        help='episode范围 (格式: start-end)')
    parser.add_argument('--delete_videos', action='store_true',
                        help='是否删除videos目录 (已自动启用)')
    
    args = parser.parse_args()
    args.delete_videos = True  # 始终删除videos目录
    
    print("开始VLN数据处理流程")
    print(f"Episode范围: {args.episode_range}")
    
    process_all_scenes(args)
    
    print("处理流程完成")

if __name__ == "__main__":
    main()
