import uuid
import os
import sys
import numpy as np
import gzip
import habitat
import cv2
import json
import argparse
import multiprocessing as mp
from tqdm import tqdm
import quaternion
from typing import Dict, List, Tuple, Any, Optional

# 常量定义
ARROW_COLOR = (0, 0, 255)  # BGR 红色
ARROW_THICKNESS = 4
VERTICAL_CROP_RATIO = 0.15  # 顶部/底部裁剪比例
ANGLE_THRESHOLD_FORWARD = 30.0
ANGLE_THRESHOLD_BACK = 90.0
TURN_ANGLE_STEP = 60.0  # 大角度转向的步长

def generate_vln_uuid(scene_type: str, scene_name: str, split: str, 
                     start_position: Any, start_rotation: Any, goal_position: Any) -> str:
    """生成唯一的VLN episode ID"""
    namespace = uuid.NAMESPACE_DNS
    key = f"{scene_type}:{scene_name}:{split}:{start_position}:{start_rotation}:{goal_position}"
    return str(uuid.uuid5(namespace, key))[:8]


def load_nav_episodes_from_gz(filepath: str) -> List[Dict]:
    """从 .json.gz 文件加载导航 episodes"""
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('episodes', [])
    except FileNotFoundError:
        print(f"错误: 找不到导航文件 {filepath}")
        return []
    except Exception as e:
        print(f"加载导航文件 {filepath} 出错: {e}")
        return []


def get_look_at_quat(pos_prev: np.ndarray, pos_curr: np.ndarray) -> List[float]:
    """计算从 pos_prev 指向 pos_curr 的偏航角(yaw)对应的四元数"""
    delta = pos_curr - pos_prev
    yaw = np.arctan2(-delta[0], -delta[2])
    
    # 转换为 (x, y, z, w) 四元数
    x = 0
    y = np.sin(yaw / 2.0)
    z = 0
    w = np.cos(yaw / 2.0)
    return [x, y, z, w]


def get_angle_description(current_pos: np.ndarray, current_rot_quat: quaternion.quaternion, 
                         next_pos: np.ndarray) -> Tuple[str, float]:
    """
    计算从当前位姿转向下一个节点所需的方向描述
    
    Returns:
        (动作描述, 角度值)
    """
    # 1. 获取当前"前向"矢量
    v_fwd_local = np.array([0, 0, -1.0])
    current_forward_vec = quaternion.rotate_vectors(current_rot_quat, v_fwd_local)
    
    # 2. 获取"目标"矢量
    target_vec = next_pos - current_pos

    # 3. 扁平化到 X-Z 地面
    v_fwd_flat = np.array([current_forward_vec[0], 0, current_forward_vec[2]])
    v_tgt_flat = np.array([target_vec[0], 0, target_vec[2]])

    # 4. 归一化
    v_fwd_norm = v_fwd_flat / (np.linalg.norm(v_fwd_flat) + 1e-8)
    v_tgt_norm = v_tgt_flat / (np.linalg.norm(v_tgt_flat) + 1e-8)

    # 5. 计算带符号的角度
    dot_product = np.dot(v_fwd_norm, v_tgt_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    cross_y = np.cross(v_fwd_norm, v_tgt_norm)[1]
    
    signed_angle_deg = np.degrees(angle_rad * np.sign(cross_y))

    # 6. 转换为动作描述
    if abs(signed_angle_deg) <= ANGLE_THRESHOLD_FORWARD:
        return "go forward", signed_angle_deg
    elif abs(signed_angle_deg) > ANGLE_THRESHOLD_BACK:
        return "go back", signed_angle_deg
    elif signed_angle_deg > ANGLE_THRESHOLD_FORWARD:
        return "go left", signed_angle_deg
    elif signed_angle_deg < -ANGLE_THRESHOLD_FORWARD:
        return "go right", signed_angle_deg
    else:
        return "go forward", signed_angle_deg


def crop_pano_image(image: np.ndarray) -> np.ndarray:
    """裁剪全景图，去除顶部/底部"""
    h, w, _ = image.shape
    top_crop = int(h * VERTICAL_CROP_RATIO)
    bottom_crop = int(h * (1.0 - VERTICAL_CROP_RATIO))
    return image[top_crop:bottom_crop, :, :]


def draw_arrow_on_image(pano_image_bgr: np.ndarray, angle_deg: float, 
                       is_final_node: bool) -> np.ndarray:
    """在全景图像上绘制方向箭头"""
    cropped_image = crop_pano_image(pano_image_bgr)
    
    if is_final_node:
        return cropped_image
    
    h_new, w_new, _ = cropped_image.shape

    # 计算箭头水平位置
    arrow_x = (w_new / 2) - (angle_deg / 360.0) * w_new
    arrow_x = int(np.clip(arrow_x, 0, w_new - 1))

    # 绘制朝上的红色小箭头
    arrow_y_bottom = h_new - 10
    arrow_y_top = h_new - 40
    
    cv2.arrowedLine(
        cropped_image, 
        (arrow_x, arrow_y_bottom), 
        (arrow_x, arrow_y_top), 
        ARROW_COLOR, 
        ARROW_THICKNESS, 
        tipLength=0.5
    )
    
    return cropped_image


def handle_large_turn(sim: habitat.Simulator, current_pos: np.ndarray, 
                     current_rot: List[float], angle_deg: float, 
                     is_left_turn: bool, episode_img_path: str, 
                     image_id: int) -> Tuple[List[str], int]:
    """
    处理大于90度的转向，分解为多个小转向
    
    Returns:
        (分解的动作列表, 新的图像ID)
    """
    actions = []
    remaining_angle = abs(angle_deg)
    turn_step = TURN_ANGLE_STEP if is_left_turn else -TURN_ANGLE_STEP
    turn_action = "go left" if is_left_turn else "go right"
    
    while remaining_angle > 0:
        # 保存当前朝向的图像
        rot_quat = quaternion.from_float_array(current_rot)
        if len(actions) > 0:  # 不是第一次，需要应用旋转
            rot_quat = rot_quat * quaternion.from_euler_angles([0, np.radians(turn_step * len(actions)), 0])
        
        try:
            observations = sim.get_observations_at(
                position=list(current_pos),
                rotation=list(quaternion.as_float_array(rot_quat)),
                keep_agent_at_new_pose=True
            )
        except Exception as e:
            print(f"  警告：获取转向观测失败: {e}")
            break
        
        # 处理并保存图像
        pano_image_rgb = observations['rgb']
        pano_image_bgr = cv2.cvtColor(pano_image_rgb, cv2.COLOR_RGB2BGR)
        current_angle = angle_deg - (turn_step * len(actions)) if is_left_turn else angle_deg + (turn_step * len(actions))
        final_image = draw_arrow_on_image(pano_image_bgr, current_angle, False)
        
        img_name = f"{image_id}.jpg"
        cv2.imwrite(os.path.join(episode_img_path, img_name), final_image)
        
        # 添加动作
        actions.append(turn_action)
        image_id += 1
        remaining_angle -= TURN_ANGLE_STEP
    
    return actions, image_id


def process_single_node(sim: habitat.Simulator, topo_node: List[float], 
                       path: List[List[float]], node_idx: int,
                       episode_img_path: str, image_id: int,
                       prev_rotation: Optional[List[float]] = None) -> Tuple[str, float, int]:
    """
    处理单个拓扑节点
    
    Returns:
        (动作描述, 角度值, 新的图像ID)
    """
    current_pos = np.array(topo_node)
    
    # 获取当前节点的朝向
    if node_idx == 0 or prev_rotation is None:
        # 第一个节点：需要从外部传入旋转
        raise ValueError("第一个节点需要提供起始旋转")
    else:
        prev_pos = np.array(path[node_idx-1])
        rot_list_xyzw = get_look_at_quat(prev_pos, current_pos)
    
    # 获取观测
    try:
        observations = sim.get_observations_at(
            position=list(current_pos),
            rotation=list(rot_list_xyzw),
            keep_agent_at_new_pose=True
        )
    except Exception as e:
        print(f"  错误：在传送至节点 {node_idx} 时失败: {e}")
        return "stop", 0.0, image_id
    
    # 计算到下一个节点的转向
    text_desc = "stop"
    angle_deg = 0.0
    is_final_node = (node_idx == len(path) - 1)
    
    if not is_final_node:
        next_pos = np.array(path[node_idx+1])
        agent_state = sim.get_agent_state()
        text_desc, angle_deg = get_angle_description(current_pos, agent_state.rotation, next_pos)
    
    # 处理图像
    pano_image_rgb = observations['rgb']
    pano_image_bgr = cv2.cvtColor(pano_image_rgb, cv2.COLOR_RGB2BGR)
    final_image = draw_arrow_on_image(pano_image_bgr, angle_deg, is_final_node)
    
    # 保存图像
    img_name = f"{image_id}.jpg"
    cv2.imwrite(os.path.join(episode_img_path, img_name), final_image)
    
    return text_desc, angle_deg, image_id + 1


def process_episode(env: habitat.Env, episode: Dict, episode_id: str, 
                   args: argparse.Namespace, batch_id: int) -> Optional[Dict]:
    """处理单个episode，生成指令和图像"""
    print(f'处理 episode: {episode_id} (批次 {batch_id})')
    
    episode_img_path = os.path.join(args.img_path, episode_id)
    os.makedirs(episode_img_path, exist_ok=True)
    
    path = episode["reference_path"]
    if not path:
        print(f"  警告：episode {episode_id} 没有参考路径")
        return None
    
    actions = []
    image_id = 1
    
    # 处理路径上的每个节点
    for n, topo_node in enumerate(path):
        if n == 0:
            # 第一个节点使用起始旋转
            current_pos = np.array(topo_node)
            try:
                observations = env.sim.get_observations_at(
                    position=list(current_pos),
                    rotation=episode["start_rotation"],
                    keep_agent_at_new_pose=True
                )
            except Exception as e:
                print(f"  错误：在传送至起始位置时失败: {e}")
                return None
            
            # 计算第一个动作
            if len(path) > 1:
                next_pos = np.array(path[1])
                agent_state = env.sim.get_agent_state()
                text_desc, angle_deg = get_angle_description(current_pos, agent_state.rotation, next_pos)
            else:
                text_desc, angle_deg = "stop", 0.0
            
            # 处理图像
            pano_image_rgb = observations['rgb']
            pano_image_bgr = cv2.cvtColor(pano_image_rgb, cv2.COLOR_RGB2BGR)
            final_image = draw_arrow_on_image(pano_image_bgr, angle_deg, len(path) == 1)
            
            img_name = f"{image_id}.jpg"
            cv2.imwrite(os.path.join(episode_img_path, img_name), final_image)
            image_id += 1
            
            # 处理大角度转向
            if angle_deg > ANGLE_THRESHOLD_BACK:
                new_actions, image_id = handle_large_turn(
                    env.sim, current_pos, episode["start_rotation"], 
                    angle_deg, True, episode_img_path, image_id
                )
                actions.extend(new_actions)
            elif angle_deg < -ANGLE_THRESHOLD_BACK:
                new_actions, image_id = handle_large_turn(
                    env.sim, current_pos, episode["start_rotation"], 
                    angle_deg, False, episode_img_path, image_id
                )
                actions.extend(new_actions)
            else:
                actions.append(text_desc)
        else:
            # 处理后续节点
            text_desc, angle_deg, new_image_id = process_single_node(
                env.sim, topo_node, path, n, episode_img_path, image_id
            )
            image_id = new_image_id
            
            # 处理大角度转向
            if angle_deg > ANGLE_THRESHOLD_BACK:
                new_actions, image_id = handle_large_turn(
                    env.sim, np.array(topo_node), get_look_at_quat(np.array(path[n-1]), np.array(topo_node)), 
                    angle_deg, True, episode_img_path, image_id
                )
                actions.extend(new_actions)
            elif angle_deg < -ANGLE_THRESHOLD_BACK:
                new_actions, image_id = handle_large_turn(
                    env.sim, np.array(topo_node), get_look_at_quat(np.array(path[n-1]), np.array(topo_node)), 
                    angle_deg, False, episode_img_path, image_id
                )
                actions.extend(new_actions)
            else:
                actions.append(text_desc)
    
    # 创建指令episode
    return {
        'episode_id': episode_id,
        'actions': actions
    }


def prepare_vln(args: argparse.Namespace, batch_size: int, batch_id: int) -> Dict:
    """准备VLN指令数据"""
    # 配置环境
    cfg = habitat.get_config("obj2vln/configs/vln_hm3d_v1.yaml")
    cfg.defrost()
    cfg.DATASET.DATA_PATH = os.path.join(args.vln_topo_path, f"{args.scene_name}.json.gz")
    cfg.freeze()

    # 加载数据
    vln_data = load_nav_episodes_from_gz(cfg.DATASET.DATA_PATH)
    if not vln_data:
        print(f"在 {cfg.DATASET.DATA_PATH} 中没有找到 episodes")
        return {'episodes': []}

    # 初始化环境
    try:
        env = habitat.Env(config=cfg)
    except Exception as e:
        print(f"初始化环境失败: {e}")
        return {'episodes': []}

    instruction_data = {'episodes': []}
    
    # 处理每个episode
    for ep_id, episode in enumerate(vln_data):
        if (ep_id % batch_size) != batch_id and batch_size > 1:
            env.reset()
            continue
        
        episode_id = generate_vln_uuid(
            args.scene_type,
            args.scene_name,
            args.split,
            episode["start_position"],
            episode["start_rotation"],
            episode["goals"][0]["position"]
        )
        
        instr_episode = process_episode(env, episode, episode_id, args, batch_id)
        if instr_episode:
            instruction_data['episodes'].append(instr_episode)
    
    env.close()
    return instruction_data


def validate_episode_range(episode_range: str) -> Tuple[int, int]:
    """验证并解析episode范围"""
    if episode_range.count('-') != 1:
        raise ValueError(f"无效的episode_range格式: {episode_range}")
    
    try:
        start_ep, end_ep = map(int, episode_range.split('-'))
    except ValueError:
        raise ValueError(f"episode_range应为整数范围: {episode_range}")
    
    if start_ep % 100 != 0 or end_ep % 100 != 99:
        raise ValueError(f"episode_range必须以100为单位划分 (例如 3100-3199): {episode_range}")
    
    return start_ep, end_ep


def get_episode_ranges(start_ep: int, end_ep: int) -> List[str]:
    """生成episode范围列表"""
    ranges = []
    for ep_start in range(start_ep, end_ep + 1, 100):
        ep_end = min(ep_start + 99, end_ep)
        ranges.append(f"episode_num_{ep_start}-{ep_end}")
    return ranges


def process_scene_range(args: argparse.Namespace, episode_range: str, batch_size: int = 6) -> None:
    """处理单个episode范围"""
    # 更新路径
    args.vln_topo_path = os.path.join(os.path.dirname(args.vln_topo_path), episode_range)
    print(f'\n处理范围: {episode_range}')
    print(f'路径: {args.vln_topo_path}')
    
    if not os.path.exists(args.vln_topo_path):
        print(f"  跳过: 路径不存在 - {args.vln_topo_path}")
        return
    
    # 创建目录
    args.img_path = os.path.join(args.vln_topo_path, 'images', args.scene_name)
    args.instruction_path = os.path.join(args.vln_topo_path, 'instructions')
    os.makedirs(args.img_path, exist_ok=True)
    os.makedirs(args.instruction_path, exist_ok=True)
    
    # 多进程处理
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=batch_size) as pool:
        instru_list = pool.starmap(
            prepare_vln, 
            [(args, batch_size, batch_id) for batch_id in range(batch_size)]
        )
        
        # 合并结果
        vln_instruction = {'episodes': []}
        for item in instru_list:
            vln_instruction['episodes'].extend(item['episodes'])
        
        # 保存结果
        json_path = os.path.join(args.instruction_path, f'{args.scene_name}.json')
        json_str = json.dumps(vln_instruction, indent=4)
        
        if args.debug == "True":
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        with gzip.open(f"{json_path}.gz", 'wt', encoding='utf-8') as gz_f:
            gz_f.write(json_str)
        
        print(f"  成功处理 {len(vln_instruction['episodes'])} 个 episodes")


def main():
    """主函数：参数解析和流程控制"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--vln_topo_path', type=str, 
                       default='data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199')
    parser.add_argument('--scene_type', type=str, default='hm3d_v1')
    parser.add_argument('--scene_name', type=str, default='2Pc8W48bu21')
    parser.add_argument('--episode_range', type=str, default='3800-3999')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', default='True', type=str, help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        start_ep, end_ep = validate_episode_range(args.episode_range)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    episode_range_list = get_episode_ranges(start_ep, end_ep)
    print(f'处理的episode范围: {episode_range_list}')
    
    for episode_range in episode_range_list:
        process_scene_range(args, episode_range)


if __name__ == "__main__":
    main()
