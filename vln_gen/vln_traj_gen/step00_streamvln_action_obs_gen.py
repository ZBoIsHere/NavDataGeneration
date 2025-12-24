import json
import sys
import os
import gzip
import math
import argparse
import uuid
import multiprocessing as mp
from PIL import Image
from scipy.spatial.transform import Rotation as R
from habitat import Env, get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    observations_to_image,
    images_to_video,
    append_text_to_image
)


def generate_vln_uuid(scene_type, scene_name, split, start_position, start_rotation, goal_position):
    """生成唯一的episode ID"""
    namespace = uuid.NAMESPACE_DNS
    key = f"{scene_type}:{scene_name}:{split}:{start_position}:{start_rotation}:{goal_position}"
    return str(uuid.uuid5(namespace, key))[:8]


def load_dataset(path):
    """安全加载gzip压缩的JSON数据集"""
    try:
        with gzip.open(path, "rb") as file:
            return json.loads(file.read(), encoding="utf-8")
    except Exception as e:
        print(f"错误: 无法加载数据集 {path}: {str(e)}")
        return {"episodes": []}


def setup_environment(scene_name, vln_topo_path):
    """初始化Habitat环境"""
    config = get_config("obj2vln/configs/vln_hm3d_v1.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = os.path.join(vln_topo_path, f'{scene_name}.json.gz')
    config.freeze()
    
    if not os.path.exists(config.DATASET.DATA_PATH):
        print(f"  警告: 数据集文件 {config.DATASET.DATA_PATH} 不存在。")
        return None
    
    dataset = load_dataset(config.DATASET.DATA_PATH)
    if not dataset.get('episodes'):
        print(f"  警告: 数据集 {scene_name} 中没有可用的 episodes。")
        return None
    
    return Env(config), dataset


def calculate_rotation_diff(agent_state, goal_position):
    """计算智能体与目标之间的旋转角度差"""
    dx = goal_position[0] - agent_state.position[0]
    dz = goal_position[2] - agent_state.position[2]
    goal_angle = -math.atan2(dx, -dz)
    goal_degree = math.degrees(goal_angle)
    
    r = R.from_quat([
        agent_state.rotation.x,
        agent_state.rotation.y,
        agent_state.rotation.z,
        agent_state.rotation.w
    ])
    agent_degree = r.as_euler('yxz', degrees=True)[0]
    
    diff = goal_degree - agent_degree
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def process_single_episode(env, follower, episode, args, scene_name, ep_id):
    """处理单个episode，生成轨迹数据和图像"""
    obs = env.reset()
    goal_position = episode.goals[0].position
    reference_path = episode.reference_path + [goal_position]
    
    # 初始化轨迹数据
    vln_traj = {
        "id": ep_id,
        "video": f'images/{scene_name}_l3_{ep_id}',
        "instructions": [],
        "actions": [-1]  # 初始动作
    }
    
    # 创建图像保存目录
    rgb_dir = os.path.join(
        args.streamvln_data_path,
        'images',
        f'{scene_name}_l3_{ep_id}',
        'rgb'
    )
    os.makedirs(rgb_dir, exist_ok=True)
    
    # 保存初始帧
    Image.fromarray(obs['rgb']).save(os.path.join(rgb_dir, '0.jpg'))
    frames = []
    step_count = 0
    
    # 跟随参考路径
    for point in reference_path:
        point = env.sim.pathfinder.snap_point(point)
        while True:
            try:
                best_action = follower.get_next_action(point)
            except Exception as e:
                print(f'获取动作错误: {e}')
                best_action = 0
            
            if best_action in (None, 0) or env.episode_over:
                break
                
            obs = env.step(best_action)
            step_count += 1
            Image.fromarray(obs['rgb']).save(os.path.join(rgb_dir, f'{step_count}.jpg'))
            
            # 保存视频帧
            frame = observations_to_image({"rgb": obs["rgb"]}, env.get_metrics())
            frame = append_text_to_image(frame, str(vln_traj['instructions']))
            frames.append(frame)
            
            vln_traj['actions'].append(best_action)
    
    # 调整最终朝向
    agent_state = env.sim.get_agent_state()
    diff = calculate_rotation_diff(agent_state, goal_position)
    
    while abs(diff) > 7.5:
        best_action = 3 if diff < 0 else 2
        diff += 15 if diff < 0 else -15
        
        try:
            obs = env.step(best_action)
        except Exception as e:
            print(f'执行动作错误: {e}')
            break
            
        step_count += 1
        Image.fromarray(obs['rgb']).save(os.path.join(rgb_dir, f'{step_count}.jpg'))
        
        frame = observations_to_image({"rgb": obs["rgb"]}, env.get_metrics())
        frame = append_text_to_image(frame, str(vln_traj['instructions']))
        frames.append(frame)
        
        vln_traj['actions'].append(best_action)
    
    return vln_traj, frames


def save_video_if_needed(frames, args, scene_name, ep_id, episode_id):
    """根据配置条件保存视频"""
    if args.save_video != "True" or episode_id % args.save_video_interval != 0:
        return
    
    video_dir = os.path.join(args.streamvln_data_path, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    images_to_video(frames, video_dir, f'{scene_name}_l3_{ep_id}.mp4')


def process_scene_batch(args, batch_size, batch_id):
    """处理场景的批量子集"""
    env_result = setup_environment(args.scene_name, args.vln_topo_path)
    if env_result is None:
        return []
    
    env, dataset = env_result
    follower = ShortestPathFollower(
        env.sim, goal_radius=0.2, return_one_hot=False, stop_on_error=False
    )
    follower.mode = "geodesic_path"
    
    max_episodes = min(args.max_episodes, len(dataset['episodes']))
    vln_dataset = []
    
    print(f'  总共 {len(dataset["episodes"])} 条 episode，处理前 {max_episodes} 条。')
    
    for episode_id in range(max_episodes):
        if batch_size > 1 and episode_id % batch_size != batch_id:
            env.reset()
            continue
            
        print(f'处理 batch {batch_id}/{batch_size}, episode {episode_id}/{max_episodes}')
        ep = dataset['episodes'][episode_id]
        ep_id = generate_vln_uuid(
            args.scene_type,
            args.scene_name,
            args.split,
            str(ep['start_position']),
            str(ep['start_rotation']),
            str(ep['goals'][0]['position']),
        )
        
        env.current_episode = ep
        vln_traj, frames = process_single_episode(
            env, follower, ep, args, args.scene_name, ep_id
        )
        save_video_if_needed(frames, args, args.scene_name, ep_id, episode_id)
        vln_dataset.append(vln_traj)
    
    env.close()
    return vln_dataset


def validate_episode_range(episode_range):
    """验证episode_range格式是否正确"""
    if episode_range.count('-') != 1:
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
    return [
        f"episode_num_{ep_start}-{min(ep_start + 99, end_ep)}"
        for ep_start in range(start_ep, end_ep + 1, 100)
    ]


def process_scene(args, scene_name):
    """处理单个场景的所有数据"""
    annotation_dir = os.path.join(args.streamvln_data_path, 'annotations')
    annotation_file = os.path.join(annotation_dir, f'{scene_name}.json.gz')
    
    # 检查是否已存在有效数据
    if os.path.exists(annotation_file):
        with gzip.open(annotation_file, "rb") as f:
            existing_data = json.loads(f.read(), encoding="utf-8")
        if len(existing_data) > 2:
            print(f"  跳过: 文件 {annotation_file} 已存在且包含非2条 episodes。")
            return
    
    # 创建必要目录
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(os.path.join(args.streamvln_data_path, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(args.streamvln_data_path, 'images'), exist_ok=True)
    
    # 多进程处理
    batch_size = min(6, mp.cpu_count())
    print(f'使用批处理大小: {batch_size}')
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=batch_size) as pool:
        results = pool.starmap(
            process_scene_batch,
            [(args, batch_size, batch_id) for batch_id in range(batch_size)]
        )
    
    # 合并结果
    vln_dataset = []
    for batch_result in results:
        vln_dataset.extend(batch_result)
    
    if not vln_dataset:
        print("  警告: 未生成任何有效数据")
        return
    
    # 保存结果
    annotation_path = os.path.join(annotation_dir, f'{scene_name}.json')
    json_str = json.dumps(vln_dataset, indent=4)
    
    if args.debug == "True":
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    with gzip.open(f"{annotation_path}.gz", 'wt', encoding='utf-8') as gz_f:
        gz_f.write(json_str)
    
    print(f"  成功处理场景 {scene_name}, 生成 {len(vln_dataset)} 条轨迹")


def process_episode_range(args, episode_range):
    """处理单个episode范围内的所有场景"""
    current_vln_path = os.path.join(os.path.dirname(args.vln_topo_path), episode_range)
    current_stream_path = os.path.join(os.path.dirname(args.streamvln_data_path), episode_range)
    
    print(f'\n处理范围: {episode_range}')
    print(f'VLN路径: {current_vln_path}')
    print(f'Stream路径: {current_stream_path}')
    
    if not os.path.exists(current_vln_path):
        print(f"  跳过: VLN路径不存在 - {current_vln_path}")
        return
    
    os.makedirs(current_stream_path, exist_ok=True)
    
    # 处理该范围内的所有场景文件
    scene_files = [f for f in os.listdir(current_vln_path) if f.endswith(".json.gz")]
    if not scene_files:
        print(f"  警告: 在 {current_vln_path} 中没有找到场景文件")
        return
    
    print(f"  发现 {len(scene_files)} 个场景文件")
    for scene_file in scene_files:
        scene_name = scene_file.replace(".json.gz", "")
        args.scene_name = scene_name
        args.vln_topo_path = current_vln_path
        args.streamvln_data_path = current_stream_path
        process_scene(args)


def main():
    """主函数：参数解析和流程控制"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vln_topo_path", type=str,
                        default="data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199")
    parser.add_argument("--streamvln_data_path", type=str,
                        default="data/traj_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199")
    parser.add_argument("--scene_type", type=str, default='hm3d_v1')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--save_video", type=str, default='True')
    parser.add_argument("--save_video_interval", type=int, default=100)
    parser.add_argument("--debug", type=str, default='True')
    parser.add_argument('--episode_range', type=str, default='3100-9999')
    parser.add_argument("--max_episodes", type=int, default=1000)
    
    args = parser.parse_args()
    
    try:
        start_ep, end_ep = validate_episode_range(args.episode_range)
    except ValueError as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    episode_ranges = get_episode_ranges(start_ep, end_ep)
    print(f'处理的episode范围: {episode_ranges}')
    
    for episode_range in episode_ranges:
        process_episode_range(args, episode_range)
    
    print("\n所有场景处理完成")


if __name__ == "__main__":
    main()
