import json
import numpy as np
import os
import sys
import gzip
import glob
import heapq
import itertools
import math
import time
import argparse
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import attr

# 常量定义
ACTION_MAP = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT',
              3: 'TURN_RIGHT', 4: 'LOOK_UP', 5: 'LOOK_DOWN'}

INSTRUCTION_VOCAB_EMPTY = {
    "word_list": [""],
    "word2idx_dict": {"": ""},
    "stoi": {"": ""},
    "itos": [""],
    "num_vocab": 0,
    "UNK_INDEX": 0,
    "PAD_INDEX": 0
}

# (必需) 转换模式 (固定为 1)
TRANSFORM_MODE = 1

# (必需) TSP 优化的航点数量上限
TSP_WAYPOINT_LIMIT = 9


def write_json(data, path):
    """将数据写入JSON文件"""
    with open(path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def write_gzip(input_path, output_path):
    """将文件压缩为gzip格式"""
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def load_dataset(path):
    """加载gzip压缩的JSON数据集"""
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def get_closest_goal(env, goal_position):
    """获取距离当前点最近的目标点"""
    sim = env.sim
    episode = env.current_episode
    min_dist = 1000.0
    goal_location = None
    agent_position = goal_position
    
    for goal in episode.goals:
        for view_point in goal.view_points:
            position = view_point.agent_state.position
            dist = sim.geodesic_distance(agent_position, position)
            if min_dist > dist:
                min_dist = dist
                goal_location = goal.position

    print(f'closest goal pos is {goal_location}, dist is {min_dist}')
    return goal_location


class ConnectivityGraph:
    """用于存储和处理单个场景的拓扑图数据"""

    def __init__(self, filepath):
        self.nodes = []
        self.image_id_to_index = {}
        self.index_to_image_id = {}
        self.original_coords = None
        self.transformed_coords = None
        self.adjacency_list = {}
        self.apsp_dist_matrix = None
        self.apsp_path_matrix = None

        try:
            self._load_nodes(filepath)
            self._build_transformed_coords(TRANSFORM_MODE)
            self._build_adjacency_list()
            self._build_all_pairs_shortest_path()
        except Exception as e:
            print(f"  [Graph Error] 构建图时出错: {e}", file=sys.stderr)
            raise

    def _load_nodes(self, filepath):
        """加载节点并建立 image_id <-> index 的映射"""
        print(f"  [Graph] 正在加载拓扑图: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        node_coords = []
        for i, node in enumerate(data):
            if 'pose' not in node or 'image_id' not in node:
                continue

            img_id = node['image_id']
            self.nodes.append(node)
            self.image_id_to_index[img_id] = i
            self.index_to_image_id[i] = img_id
            node_coords.append(
                [node['pose'][3], node['pose'][7], node['pose'][11]])

        self.original_coords = np.array(node_coords)
        print(f"  [Graph] 加载了 {len(self.nodes)} 个节点。")

    def _build_transformed_coords(self, mode):
        """根据模式1转换坐标 (xn=xc, yn=zc, zn=-yc)"""
        if mode != 1:
            print("  [Graph Warning] 此脚本被配置为只使用模式1。")

        tx_coords = self.original_coords[:, [0, 2, 1]]
        tx_coords[:, 2] *= -1
        self.transformed_coords = tx_coords

    def _build_adjacency_list(self):
        """构建邻接表，使用原始坐标计算欧氏距离"""
        total_nodes = len(self.nodes)
        for i, node in enumerate(self.nodes):
            self.adjacency_list[i] = []
            if 'unobstructed' not in node:
                continue

            for j, is_unobstructed in enumerate(node['unobstructed']):
                if is_unobstructed and j < total_nodes:
                    dist = np.linalg.norm(
                        self.original_coords[i] - self.original_coords[j])
                    self.adjacency_list[i].append((j, dist))

    def _find_shortest_paths_from_source(self, start_index):
        """(Dijkstra) 运行从单个源到所有其他节点"""
        distances = {node: np.inf for node in range(len(self.nodes))}
        paths = {node: None for node in range(len(self.nodes))}

        pq = [(0, start_index, [start_index])]
        distances[start_index] = 0
        paths[start_index] = [start_index]
        visited = set()

        while pq:
            (dist, current_idx, path) = heapq.heappop(pq)

            if current_idx in visited:
                continue
            visited.add(current_idx)

            if current_idx not in self.adjacency_list:
                continue

            for neighbor_idx, weight in self.adjacency_list[current_idx]:
                if neighbor_idx not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor_idx]:
                        distances[neighbor_idx] = new_dist
                        new_path = path + [neighbor_idx]
                        paths[neighbor_idx] = new_path
                        heapq.heappush(pq, (new_dist, neighbor_idx, new_path))

        return distances, paths

    def _build_all_pairs_shortest_path(self):
        """预计算所有节点对之间的最短路径和距离"""
        print("  [Graph] 正在预计算所有节点对的最短路径 (APSP)...")
        n = len(self.nodes)
        self.apsp_dist_matrix = np.full((n, n), np.inf)
        self.apsp_path_matrix = [[None for _ in range(n)] for _ in range(n)]

        for i in tqdm(range(n), desc="  [Graph] APSP 计算进度", unit="node", leave=False):
            distances, paths = self._find_shortest_paths_from_source(i)
            for j in range(n):
                self.apsp_dist_matrix[i, j] = distances[j]
                self.apsp_path_matrix[i][j] = paths[j]
        print("  [Graph] APSP 预计算完毕。")

    def find_nearest_node(self, nav_point_np):
        """为单个导航点 (Y-up) 找到最近的拓扑图节点"""
        if nav_point_np.ndim == 1:
            nav_point_np = nav_point_np.reshape(1, 3)

        dist_matrix = cdist(nav_point_np, self.transformed_coords, 'euclidean')
        nn_index = dist_matrix.argmin()
        min_dist = dist_matrix[0, nn_index]
        image_id = self.index_to_image_id[nn_index]

        return nn_index, image_id, min_dist

    def get_path_from_indices(self, path_indices):
        """将索引列表转换为 image_id 列表"""
        return [self.index_to_image_id[idx] for idx in path_indices]

    def get_path_distance(self, path_indices):
        """计算索引路径的总欧氏距离"""
        dist = 0
        for i in range(len(path_indices) - 1):
            segment_dist = self.apsp_dist_matrix[path_indices[i], path_indices[i + 1]]
            if segment_dist == np.inf:
                return np.inf
            dist += segment_dist
        return dist

    def get_original_coords_from_indices(self, path_indices):
        """获取路径的原始 (xc, yc, zc) 坐标"""
        return self.original_coords[path_indices]

    def _solve_tsp_permutation(self, start_node, stop_node, waypoints_to_perm):
        """TSP 排列求解的辅助函数"""
        min_cost = np.inf
        best_sequence = None

        for perm in itertools.permutations(waypoints_to_perm):
            current_sequence = [start_node] + list(perm) + [stop_node]
            current_cost = 0
            valid_perm = True

            for k in range(len(current_sequence) - 1):
                cost_segment = self.apsp_dist_matrix[current_sequence[k], current_sequence[k + 1]]
                if cost_segment == np.inf:
                    valid_perm = False
                    break
                current_cost += cost_segment

            if valid_perm and current_cost < min_cost:
                min_cost = current_cost
                best_sequence = current_sequence

        return min_cost, best_sequence


def extract_nav_points(episode, env):
    """从episode中提取导航点"""
    nav_points_to_match = [episode.start_position]
    reference_replay = episode.reference_replay
    
    # 跳过第一个STOP动作
    if reference_replay and reference_replay[0]['action'] == "STOP":
        reference_replay = reference_replay[1:]
    
    # 提取移动点
    for step in reference_replay:
        action = step.get('action')
        if action in ("MOVE_FORWARD", "STOP"):
            nav_points_to_match.append(step['agent_state']['position'])
    
    # 获取最近的目标点
    goal_position = reference_replay[-1]['agent_state']['position'] if reference_replay else episode.goals[0].position
    goal_position = get_closest_goal(env, goal_position)
    
    return nav_points_to_match, goal_position


def match_nav_points_to_graph(graph, nav_points):
    """将导航点匹配到拓扑图节点"""
    preliminary_indices = []
    match_errors = []

    for point in nav_points:
        idx, img_id, dist = graph.find_nearest_node(np.array(point))
        preliminary_indices.append(idx)
        match_errors.append(dist)
    
    return preliminary_indices, match_errors


def deduplicate_path_indices(preliminary_indices, graph, dist_threshold):
    """去重路径索引，移除相邻重复和空间上接近的点"""
    if not preliminary_indices:
        return [], []

    # 只去除相邻重复
    deduplicated_indices = [preliminary_indices[0]]
    for i in range(1, len(preliminary_indices)):
        if preliminary_indices[i] != preliminary_indices[i - 1]:
            deduplicated_indices.append(preliminary_indices[i])

    # 完全去重 + 距离阈值过滤
    fully_deduplicated_indices = [preliminary_indices[0]]
    for i in range(1, len(preliminary_indices)):
        current_idx = preliminary_indices[i]
        
        # 检查是否已存在
        if current_idx in fully_deduplicated_indices:
            continue
        
        # 检查与之前所有点的距离
        too_close = False
        for j in range(len(fully_deduplicated_indices)):
            prev_idx = fully_deduplicated_indices[j]
            if prev_idx != current_idx:
                dist_ij = graph.apsp_dist_matrix[current_idx, prev_idx]
                if dist_ij < dist_threshold:
                    too_close = True
                    print(f"--------REMOVE NEARBY POINT--------, dist is {dist_ij}")
                    break
        
        if not too_close:
            fully_deduplicated_indices.append(current_idx)
        
        # 如果当前点是终点，停止添加后续点
        if current_idx == preliminary_indices[-1]:
            print(f"--------BREAK AT GOAL POINT--------")
            break

    return deduplicated_indices, fully_deduplicated_indices


def optimize_path(graph, fully_deduplicated_indices):
    """优化路径（当前直接返回去重后的路径）"""
    # 当前实现直接返回去重后的路径，保留TSP优化框架
    return fully_deduplicated_indices, graph.get_path_distance(fully_deduplicated_indices)


def execute_path_in_env(env, follower, reference_path, goal_position):
    """在环境中执行优化后的路径"""
    original_replay = env.current_episode.reference_replay.copy()
    env.current_episode.reference_replay = []
    
    # 添加初始STOP动作
    env.current_episode.reference_replay.append({'action': ACTION_MAP[0], 'agent_state': {}})
    
    is_found = False
    best_action = 0
    
    # 执行路径
    for point in reference_path:
        if is_found:
            break
        
        done = False
        snapped_point = env.sim.pathfinder.snap_point(point)
        
        while not done and not is_found:
            try:
                best_action = follower.get_next_action(snapped_point)
            except Exception as e:
                print(f'get_next_action error: {e}')
                best_action = 0
            
            # 检查是否到达目标
            dist = env.get_metrics()['distance_to_goal']
            if dist < 0.075:
                print(f'Arrive at goal, dist is {dist}')
                is_found = True
                best_action = 0
                break

            if best_action in (None, 0) or env.episode_over:
                done = True
                continue
            
            env.step(best_action)
            env.current_episode.reference_replay.append({
                'action': ACTION_MAP[best_action],
                'agent_state': {}
            })
    
    # 调整朝向
    agent_state = env.sim.get_agent_state()
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
    
    print(f'diff deg is {diff}')
    while abs(diff) > 15:
        best_action = 3 if diff < 0 else 2
        diff += 30 if diff < 0 else -30
        
        env.current_episode.reference_replay.append({
            'action': ACTION_MAP[best_action],
            'agent_state': {}
        })
    
    # 添加最终STOP动作
    env.current_episode.reference_replay.append({'action': ACTION_MAP[0], 'agent_state': {}})
    
    # 验证执行结果
    metrics = env.get_metrics()
    success = (
        metrics.get('distance_to_goal', float('inf')) <= 0.1 and
        metrics.get('success', 0) >= 0.9 and
        0.1 <= metrics.get('spl', 0) <= 0.9 and
        len(env.current_episode.reference_replay) < 500
    )
    
    if not success:
        print(f'未能成功到达目标 (距离: {metrics.get("distance_to_goal"):.2f} 米), '
              f'成功率: {metrics.get("success")}, SPL: {metrics.get("spl"):.2f}, '
              f'路径长度: {len(env.current_episode.reference_replay)}')
        env.current_episode.reference_replay = original_replay
    
    return success


def create_vln_episode(episode, optimized_nodes, goal_position):
    """创建VLN格式的episode"""
    opt_coords_zup = optimized_nodes
    opt_coords_yup = opt_coords_zup[:, [0, 2, 1]]
    opt_coords_yup[:, 2] *= -1
    
    return {
        "episode_id": episode.episode_id,
        "trajectory_id": "",
        "scene_id": episode.scene_id,
        "start_position": episode.start_position,
        "start_rotation": episode.start_rotation,
        "instruction": {
            "instruction_text": "",
            "instruction_tokens": [],
        },
        "goals": [{
            'position': goal_position,
            'radius': 2,
        }],
        'reference_path': opt_coords_yup.tolist(),
        'info': episode.info,
    }


def process_single_episode(env, episode, graph, follower, args):
    """处理单个episode"""
    if not episode.reference_replay:
        return None, None, True  # 无轨迹，标记为不可通行
    
    # 1. 提取导航点
    nav_points, goal_position = extract_nav_points(episode, env)
    
    # 2. 匹配到拓扑图
    preliminary_indices, match_errors = match_nav_points_to_graph(graph, nav_points)
    
    if len(preliminary_indices) < 2:
        return None, match_errors, True
    
    # 3. 去重
    deduplicated_indices, fully_deduplicated_indices = deduplicate_path_indices(
        preliminary_indices, graph, args.dist_threshold
    )
    
    if len(fully_deduplicated_indices) < 2:
        return None, match_errors, True
    
    # 4. 优化路径
    optimized_indices, _ = optimize_path(graph, fully_deduplicated_indices)
    
    # 5. 获取优化后的坐标
    optimized_nodes = graph.get_original_coords_from_indices(optimized_indices)
    
    # 6. 转换坐标系 (Z-up -> Y-up)
    opt_coords_zup = optimized_nodes
    opt_coords_yup = opt_coords_zup[:, [0, 2, 1]]  # [xc, zc, yc]
    opt_coords_yup[:, 2] *= -1  # [xc, zc, -yc]
    
    # 7. 执行路径并验证
    reference_path = opt_coords_yup.tolist()
    reference_path.append([goal_position[0], goal_position[1], goal_position[2]])
    
    success = execute_path_in_env(env, follower, reference_path, goal_position)
    
    # 8. 创建VLN episode
    vln_episode = create_vln_episode(episode, optimized_nodes, goal_position)
    
    # 9. 准备ObjectNav数据
    episode_json = attr.asdict(episode)
    del episode_json['_shortest_path_cache']
    episode_json['goals'] = []
    
    return vln_episode, match_errors, success


def process_scene(args, batch_size, batch_id):
    """处理单个场景的所有episode"""
    print(f'批次 {batch_id} 正在处理...')
    scene_name = args.scene_name
    input_path = args.input_path
    topo_path = args.topo_path

    scene_file = os.path.join(input_path, scene_name + '.json.gz')
    conn_pattern = os.path.join(topo_path, f"*{scene_name}_connectivity.json")
    conn_files_found = glob.glob(conn_pattern)

    if not conn_files_found:
        print(f"--- 正在处理 {scene_name} ---")
        print(f"  错误: 找不到匹配的拓扑图文件 (模式: {conn_pattern})")
        print("-" * 60)
        return None

    full_topo_path = conn_files_found[0]
    print(f"--- 正在处理 {scene_name} ---")

    try:
        graph = ConnectivityGraph(full_topo_path)
    except Exception:
        print(f"  错误: 无法加载或构建拓扑图 {full_topo_path}。跳过此场景。")
        print("-" * 60)
        return None

    # 配置环境
    config = habitat.get_config(
        config_paths="NavTrajSampleGeneration/L3MVN/envs/habitat/configs/tasks/objectnav_hm3d_v2_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.DATASET.DATA_PATH = scene_file
    config.freeze()

    with habitat.Env(config) as env:
        dataset = load_dataset(config.DATASET.DATA_PATH)
        dataset['episodes'] = []
        follower = ShortestPathFollower(env.sim, goal_radius=0.2, return_one_hot=False, stop_on_error=False)
        follower.mode = "geodesic_path"

        vln_episodes = []
        scene_match_errors = []
        unpassable_episodes = 0
        total_episodes = min(args.max_episodes, len(env.episodes))
        
        print(f'  总共 {len(env.episodes)} 条 episode，处理前 {total_episodes} 条。')

        # 处理每个episode
        for ep_idx in range(total_episodes):
            if (ep_idx % batch_size) != batch_id and batch_size > 1:
                env.reset()
                continue
                
            env.reset()
            print(f'处理批次 {batch_id} 中的 episode {ep_idx}')
            episode = env.current_episode
            
            vln_ep, match_errors, success = process_single_episode(
                env, episode, graph, follower, args
            )
            
            if match_errors:
                scene_match_errors.extend(match_errors)
            
            if vln_ep:
                vln_episodes.append(vln_ep)
                episode_json = attr.asdict(episode)
                del episode_json['_shortest_path_cache']
                episode_json['goals'] = []
                dataset['episodes'].append(episode_json)
            else:
                unpassable_episodes += 1
                episode_json = attr.asdict(episode)
                del episode_json['_shortest_path_cache']
                episode_json['goals'] = []
                dataset['episodes'].append(episode_json)

    vln_data = {
        "instruction_vocab": INSTRUCTION_VOCAB_EMPTY,
        "episodes": vln_episodes
    }

    return dataset


def main():
    """主函数：参数解析和流程控制"""
    parser = argparse.ArgumentParser(description="transfer objectnav traj to topo node and refine")
    parser.add_argument('--input_path', type=str, required=False, 
                       default="data/traj_datasets/objectnav/hm3d_v2_hd_hd_30K_samewithL3_test/train/content",
                       help='objectnav轨迹所在的文件夹路径')
    parser.add_argument('--topo_path', type=str, required=False, 
                       default="data/scene_datasets/r2r_preprocess_data/connectivity",
                       help='scene topo graph path')
    parser.add_argument("--scene_name", type=str, default='1S7LAXRdDqK', help="scene name")
    parser.add_argument("--max_episodes", type=int, default=10, help="max episodes to process")
    parser.add_argument('--output_path', type=str, required=False,
                       default="data/traj_datasets/objectnav/hm3d_v1_hd_l3mvn_refine_v100_30k/train/content_topo",
                       help='vln轨迹所在的文件夹路径')
    parser.add_argument('--refine_path', type=str, required=False,
                       default="data/traj_datasets/objectnav/hm3d_v1_hd_l3mvn_refine_v100_30k/train/content2",
                       help='vln轨迹所在的文件夹路径')
    parser.add_argument('--dist_threshold', type=float, required=False, default=0.75,
                       help='dist threshold to consider as duplicate')

    args = parser.parse_args()

    # 验证路径
    if not os.path.isdir(args.input_path):
        print(f"错误: 输入路径 '{args.input_path}' 不是一个有效的文件夹。")
        sys.exit(1)

    if not os.path.isdir(args.topo_path):
        print(f"错误: 拓扑图路径 '{args.topo_path}' 不是一个有效的文件夹。")
        sys.exit(1)

    scene_file = os.path.join(args.input_path, args.scene_name + '.json.gz')
    if not os.path.isfile(scene_file):
        print(f"{scene_file} not exist")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.refine_path, exist_ok=True)

    if args.input_path == args.output_path:
        print(f'{args.input_path} can not equal to {args.output_path}')
        sys.exit(1)

    # 多进程处理
    batch_size = 6
    print(f'使用批处理大小: {batch_size}')
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()
    with Pool(processes=batch_size) as pool:
        results = pool.starmap(process_scene, [(args, batch_size, batch_id) for batch_id in range(batch_size)])
        batch_process_time = time.time() - start_time
        print(f'批处理时间: {batch_process_time:.2f} 秒')

        # 合并结果
        if not results or not results[0]:
            print("未生成有效结果")
            return
            
        combined_dataset = results[0]
        for dataset in results[1:]:
            if dataset:
                combined_dataset['episodes'].extend(dataset['episodes'])
        
        # 保存结果
        json_path = os.path.join(args.refine_path, f'{args.scene_name}.json')
        json_str = json.dumps(combined_dataset)
        with gzip.open(f"{json_path}.gz", 'wt', encoding='utf-8') as gz_f:
            gz_f.write(json_str)
        
        end_time = time.time()
        print(f'总处理时间: {end_time - start_time:.2f} 秒')
        print(f'结果已保存至: {json_path}.gz')


if __name__ == "__main__":
    main()
