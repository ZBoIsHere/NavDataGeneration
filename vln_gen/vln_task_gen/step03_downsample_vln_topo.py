import os
import json
import gzip
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 常量定义
MIN_DISTANCE_THRESHOLD = 0.75  # 点间最小距离阈值
FINAL_POINT_THRESHOLD = 0.5    # 终点保留阈值
MAX_PATH_LENGTH = 100          # 超过此长度的路径不简化


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    计算两个3D点之间的欧几里得距离
    
    Args:
        point1: 第一个点的坐标 [x, y, z]
        point2: 第二个点的坐标 [x, y, z]
    
    Returns:
        两点之间的距离
    """
    return (
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2
    ) ** 0.5


def filter_middle_points(points: List[List[float]], 
                        existing_points: List[List[float]]) -> List[List[float]]:
    """
    过滤中间点，保留与已有点距离大于阈值的点
    
    Args:
        points: 候选中间点列表
        existing_points: 已经保留的点列表
    
    Returns:
        过滤后的点列表
    """
    filtered_points = []
    
    for point in points:
        keep_point = True
        
        # 检查与所有已保留点的距离
        for kept_point in existing_points:
            if calculate_distance(point, kept_point) < MIN_DISTANCE_THRESHOLD:
                keep_point = False
                break
                
        if keep_point:
            filtered_points.append(point)
            existing_points.append(point)
            
    return filtered_points


def handle_final_point(final_point: List[float], 
                      last_kept_point: List[float]) -> Tuple[List[List[float]], bool]:
    """
    处理路径的最后一个点
    
    Args:
        final_point: 路径的最后一个点
        last_kept_point: 当前最后一个保留的点
    
    Returns:
        (包含最终点的列表, 是否替换最后一个点)
    """
    dist = calculate_distance(final_point, last_kept_point)
    
    if dist >= FINAL_POINT_THRESHOLD:
        return [final_point], False
    else:
        return [final_point], True  # 替换最后一个点


def simplify_reference_path(reference_path: List[List[float]]) -> List[List[float]]:
    """
    简化参考路径，保留关键点
    
    简化策略：
    1. 保留第一个点
    2. 过滤中间点，只保留与已保留点距离大于阈值的点
    3. 根据距离阈值决定是否保留最后一个点
    
    Args:
        reference_path: 原始参考路径，格式为[[x,y,z],...]
    
    Returns:
        简化后的参考路径
    """
    if len(reference_path) <= 2:
        return reference_path.copy()
    
    # 1. 保留第一个点
    simplified_path = [reference_path[0]]
    
    # 2. 处理中间点
    middle_points = reference_path[1:-1]
    filter_middle_points(middle_points, simplified_path)
    
    # 3. 处理最后一个点
    final_point = reference_path[-1]
    last_kept_point = simplified_path[-1]
    final_points, replace_last = handle_final_point(final_point, last_kept_point)
    
    if replace_last:
        simplified_path[-1] = final_points[0]
    else:
        simplified_path.extend(final_points)
    
    return simplified_path


def load_scene_data(file_path: str) -> Dict[str, Any]:
    """
    加载场景数据
    
    Args:
        file_path: 场景文件路径（.json.gz格式）
    
    Returns:
        解析后的JSON数据
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析失败
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"无法加载场景文件 {file_path}: {str(e)}")


def save_simplified_data(data: Dict[str, Any], output_path: str):
    """
    保存简化后的场景数据
    
    Args:
        data: 简化后的数据
        output_path: 输出路径（不含扩展名）
    """
    # 保存JSON文件
    json_path = f"{output_path}.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"  已保存JSON文件: {json_path}")
    except Exception as e:
        raise RuntimeError(f"无法保存JSON文件 {json_path}: {str(e)}")
    
    # 保存GZ压缩文件
    gz_path = f"{output_path}.json.gz"
    try:
        with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"  已保存GZ文件: {gz_path}")
    except Exception as e:
        # 如果GZ保存失败，删除已创建的JSON文件
        if os.path.exists(json_path):
            os.remove(json_path)
        raise RuntimeError(f"无法保存GZ文件 {gz_path}: {str(e)}")


def process_single_scene(input_path: str, output_dir: str, scene_name: str) -> int:
    """
    处理单个场景的简化
    
    Args:
        input_path: 输入目录路径
        output_dir: 输出目录路径
        scene_name: 场景名称
    
    Returns:
        处理的episode数量
    """
    # 构建文件路径
    input_file = os.path.join(input_path, f"{scene_name}.json.gz")
    output_base = os.path.join(output_dir, scene_name)
    
    print(f"处理场景: {scene_name}")
    
    # 验证输入文件
    if not os.path.exists(input_file):
        print(f"  跳过: 输入文件不存在 - {input_file}")
        return 0
    
    # 加载场景数据
    try:
        scene_data = load_scene_data(input_file)
    except Exception as e:
        print(f"  错误: {str(e)}")
        return 0
    
    # 创建结果结构
    result = {
        "episodes": [],
        "instruction_vocab": scene_data.get("instruction_vocab", {})
    }
    
    processed_count = 0
    skipped_count = 0
    
    # 处理每个episode
    for episode in scene_data.get("episodes", []):
        reference_path = episode.get("reference_path", [])
        
        # 跳过过长的路径
        if len(reference_path) > MAX_PATH_LENGTH:
            skipped_count += 1
            continue
        
        # 简化路径
        simplified_path = simplify_reference_path(reference_path)
        episode["reference_path"] = simplified_path
        processed_count += 1
        
        # 打印长度信息
        print(f"  Episode {episode.get('episode_id', 'N/A')}: "
              f"原始长度={len(reference_path)}, 简化后长度={len(simplified_path)}")
        
        result["episodes"].append(episode)
    
    # 保存结果
    save_simplified_data(result, output_base)
    
    print(f"  总结: 处理 {processed_count} 个episodes, 跳过 {skipped_count} 个过长路径")
    return processed_count


def get_scene_files(directory: str) -> List[str]:
    """
    获取目录中所有场景文件名（不含扩展名）
    
    Args:
        directory: 目录路径
    
    Returns:
        场景名称列表
    """
    scene_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".json.gz"):
            scene_names.append(file_name.replace(".json.gz", ""))
    return scene_names


def process_directory(input_base: str, output_base: str, episode_range: str) -> int:
    """
    处理指定episode范围内的所有场景
    
    Args:
        input_base: 输入基础目录
        output_base: 输出基础目录
        episode_range: episode范围目录名（如"episode_num_3100-3199"）
    
    Returns:
        总共处理的episodes数量
    """
    input_dir = os.path.join(input_base, episode_range)
    output_dir = os.path.join(output_base, episode_range)
    
    print(f"\n处理目录: {episode_range}")
    print(f"  输入路径: {input_dir}")
    print(f"  输出路径: {output_dir}")
    
    # 验证输入目录
    if not os.path.exists(input_dir):
        print(f"  跳过: 输入目录不存在 - {input_dir}")
        return 0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取场景列表
    scene_names = get_scene_files(input_dir)
    if not scene_names:
        print("  警告: 未找到任何场景文件")
        return 0
    
    print(f"  共找到 {len(scene_names)} 个场景文件")
    
    # 处理每个场景
    total_processed = 0
    for scene_name in scene_names:
        processed = process_single_scene(input_dir, output_dir, scene_name)
        total_processed += processed
    
    return total_processed


def validate_paths(input_path: str, output_path: str):
    """
    验证输入和输出路径
    
    Args:
        input_path: 输入路径
        output_path: 输出路径
    
    Raises:
        ValueError: 路径无效或相同
    """
    if not os.path.exists(input_path):
        raise ValueError(f"输入路径不存在: {input_path}")
    
    if input_path == output_path:
        raise ValueError("输入路径和输出路径不能相同")
    
    # 确保输入路径是目录
    if not os.path.isdir(input_path):
        raise ValueError(f"输入路径必须是目录: {input_path}")


def main():
    """主函数：解析参数并启动处理流程"""
    parser = argparse.ArgumentParser(description='简化VLN拓扑路径')
    parser.add_argument("--vln_topo_path", type=str, 
                        default='data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2/train/content',
                        help="原始VLN拓扑数据基础路径")
    parser.add_argument("--vln_topo_simplified_path", type=str, 
                        default='data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content',
                        help="简化后的VLN拓扑数据保存路径")
    parser.add_argument("--episode_ranges", nargs='+', 
                        default=['episode_num_3100-3199'],
                        help="要处理的episode范围列表")
    
    args = parser.parse_args()
    
    try:
        validate_paths(args.vln_topo_path, args.vln_topo_simplified_path)
    except ValueError as e:
        print(f"路径验证错误: {str(e)}")
        return
    
    print("=" * 60)
    print("VLN拓扑路径简化工具")
    print(f"输入基础路径: {args.vln_topo_path}")
    print(f"输出基础路径: {args.vln_topo_simplified_path}")
    print(f"处理范围: {args.episode_ranges}")
    print("=" * 60)
    
    total_episodes = 0
    for episode_range in args.episode_ranges:
        processed = process_directory(
            args.vln_topo_path,
            args.vln_topo_simplified_path,
            episode_range
        )
        total_episodes += processed
    
    print("\n" + "=" * 60)
    print(f"处理完成! 总共处理了 {total_episodes} 个episodes")
    print("=" * 60)


if __name__ == "__main__":
    main()
