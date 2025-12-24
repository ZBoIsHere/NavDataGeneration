import base64
import os
import sys
import time
import gzip
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any

import dashscope
from dashscope import MultiModalConversation

from gemini_annotation_prompt import (
    viewpoint_template,
    direction_template,
    remaining_prompt
)


def image_to_base64(image_path: str) -> str:
    """将图片转为 base64 格式"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        raise RuntimeError(f"图片编码失败 {image_path}: {str(e)}")


def load_instructions(file_path: str) -> Dict:
    """加载已有的指令数据"""
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"加载指令文件失败 {file_path}: {str(e)}")


def save_instructions(data: Dict, file_path: str) -> None:
    """保存指令数据到gzip压缩文件"""
    try:
        json_str = json.dumps(data, indent=4)
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            f.write(json_str)
    except Exception as e:
        raise RuntimeError(f"保存指令文件失败 {file_path}: {str(e)}")


def should_skip_episode(episode: Dict, img_dir: str, max_images: int = 50) -> bool:
    """检查是否应该跳过此episode"""
    episode_id = episode["episode_id"]
    
    # 检查是否已有指令
    if 'instruction' in episode and episode['instruction'].strip() != '':
        print(f"  跳过: episode {episode_id} 已有指令")
        return True
    
    # 检查图片数量
    img_path = os.path.join(img_dir, episode_id)
    if not os.path.exists(img_path):
        print(f"  跳过: episode {episode_id} 图片目录不存在: {img_path}")
        return True
    
    img_files = [f for f in os.listdir(img_path) if f.endswith(".jpg")]
    if len(img_files) > max_images:
        print(f"  跳过: episode {episode_id} 图片数量 ({len(img_files)}) 超过限制 {max_images}")
        return True
    
    return False


def build_messages(episode: Dict, img_dir: str) -> List[Dict]:
    """构建发送给Qwen-VL的消息"""
    episode_id = episode["episode_id"]
    img_path = os.path.join(img_dir, episode_id)
    img_files = sorted(
        [f for f in os.listdir(img_path) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0])
    )
    
    messages = [{"role": "user", "content": []}]
    content = messages[0]["content"]
    
    # 添加专家角色描述
    content.append({"text": (
        "You are an expert in object navigation and vision-language navigation. "
        "You have been provided with a sequence of viewpoints along a navigation path, "
        "represented by images and corresponding action descriptions. "
        "Your task is to generate clear and concise navigation instructions based on these images and actions. "
        "The format of the instructions should be similar to those found in R2R and RxR datasets."
    )})
    
    content.append({"text": "Here are the viewpoints and actions:"})
    
    # 添加每个视角
    for i, img_file in enumerate(img_files):
        # 添加视角文本
        content.append({"text": viewpoint_template.format(p=i)})
        
        # 添加图片
        full_img_path = os.path.join(img_path, img_file)
        try:
            base64_image = image_to_base64(full_img_path)
            content.append({"image": base64_image})
        except Exception as e:
            raise RuntimeError(f"处理图片失败 {full_img_path}: {str(e)}")
        
        # 添加动作描述（如果不是最后一张）
        if i + 1 < len(img_files) and i < len(episode["actions"]):
            action_text = episode["actions"][i]
            content.append({"text": direction_template.format(action=action_text, p=i+1)})
    
    # 添加最后的总结指令
    content.append({"text": remaining_prompt.format(view_num=len(img_files))})
    
    return messages


def call_qwen_vl_api(messages: List[Dict], episode_id: str) -> str:
    """调用Qwen-VL API并处理响应"""
    try:
        response = MultiModalConversation.call(
            model="qwen3-vl-flash",
            messages=messages
        )
    except Exception as e:
        raise RuntimeError(f"API调用失败 [{episode_id}]: {str(e)}")
    
    # 检查HTTP错误
    if response.status_code != 200:
        request_id = getattr(response, "request_id", "N/A")
        msg = getattr(response, "message", "No message")
        error_msg = f"[{response.status_code}] {msg} (Request ID: {request_id})"
        
        if "Exceeded limit on max data-uri per request" in error_msg:
            raise RuntimeError(f"请求数据超限 [{episode_id}]: {error_msg}")
        elif 'Input data may contain inappropriate content' in error_msg:
            raise RuntimeError(f"内容违规 [{episode_id}]: {error_msg}")
        elif 'Allocated quota exceeded' in error_msg:
            raise RuntimeError(f"配额超限 [{episode_id}]: {error_msg}")
        else:
            raise RuntimeError(f"API错误 [{episode_id}]: {error_msg}")
    
    # 检查响应内容
    if not response.output or not response.output.choices:
        raise RuntimeError(f"空响应 [{episode_id}]")
    
    # 提取文本内容
    choice = response.output.choices[0]
    content = choice.message.content
    
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        result = "\n".join(texts).strip()
    elif isinstance(content, str):
        result = content.strip()
    else:
        result = str(content).strip()
    
    if not result:
        raise RuntimeError(f"空指令文本 [{episode_id}]")
    
    return result.lstrip("Instruction: ")


def process_episode(episode: Dict, img_dir: str, episode_id: str) -> Optional[str]:
    """处理单个episode，生成指令"""
    start_time = time.time()
    print(f"  处理 episode: {episode_id}")
    
    # 构建消息
    try:
        messages = build_messages(episode, img_dir)
    except Exception as e:
        print(f"  构建消息失败: {str(e)}")
        return None
    
    # 调用API
    try:
        instruction = call_qwen_vl_api(messages, episode_id)
        print(f"  生成指令: {instruction}")
        return instruction
    except Exception as e:
        print(f"  生成指令失败: {str(e)}")
        return None
    finally:
        end_time = time.time()
        print(f"  处理耗时: {end_time - start_time:.2f} 秒")


def process_scene(args: argparse.Namespace) -> None:
    """处理单个场景的所有episodes"""
    print(f"\n处理场景: {args.scene_name}")
    
    # 构建路径
    instructions_file = os.path.join(args.instructions_path, f"{args.scene_name}.json.gz")
    img_dir = os.path.join(args.img_path, args.scene_name)
    
    # 验证目录
    if not os.path.exists(img_dir):
        print(f"  跳过: 图片目录不存在 - {img_dir}")
        return
    
    # 加载指令数据
    try:
        instructions_data = load_instructions(instructions_file)
        output_data = instructions_data.copy()
    except Exception as e:
        print(f"  跳过: {str(e)}")
        return
    
    # 处理每个episode
    save_counter = 0
    processed_count = 0
    skipped_count = 0
    
    for episode in instructions_data["episodes"]:
        episode_id = episode["episode_id"]
        
        # 检查是否跳过
        if should_skip_episode(episode, img_dir):
            skipped_count += 1
            continue
        
        # 生成指令
        instruction = process_episode(episode, img_dir, episode_id)
        if instruction:
            episode["instruction"] = instruction
            
            # 更新输出数据
            for out_ep in output_data["episodes"]:
                if out_ep["episode_id"] == episode_id:
                    out_ep["instruction"] = instruction
                    break
            
            processed_count += 1
            save_counter += 1
            
            # 定期保存进度
            if save_counter >= 5:
                save_instructions(output_data, instructions_file)
                print(f"  已保存进度，处理了 {processed_count} 个episodes")
                save_counter = 0
    
    # 最终保存
    save_instructions(output_data, instructions_file)
    print(f"  完成: 处理 {processed_count} 个, 跳过 {skipped_count} 个 episodes")


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


def get_scene_files(directory: str) -> List[str]:
    """获取目录中所有场景文件"""
    return [f.replace(".json.gz", "") 
            for f in os.listdir(directory) 
            if f.endswith(".json.gz")]


def main():
    """主函数：参数解析和流程控制"""
    parser = argparse.ArgumentParser(description='VLN指令生成器')
    parser.add_argument("--vln_topo_path", type=str,
                       default="data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199",
                       help="VLN拓扑数据路径")
    parser.add_argument('--episode_range', type=str, default='3100-3499',
                       help="episode范围 (格式: start-end)")
    parser.add_argument("--scene_type", type=str, default="hm3d_v1",
                       help="场景类型")
    parser.add_argument("--split", type=str, default="train",
                       help="数据集划分")
    parser.add_argument("--api_key", type=str, required=True,
                       help="Dashscope API Key")
    args = parser.parse_args()
    
    # 设置API密钥
    dashscope.api_key = args.api_key
    
    # 验证并分割episode范围
    try:
        start_ep, end_ep = validate_episode_range(args.episode_range)
    except ValueError as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    episode_ranges = [
        f"episode_num_{ep_start}-{min(ep_start + 99, end_ep)}"
        for ep_start in range(start_ep, end_ep + 1, 100)
    ]
    
    print(f"处理的episode范围: {episode_ranges}")
    
    # 处理每个范围
    for episode_range in episode_ranges:
        base_path = os.path.dirname(args.vln_topo_path)
        current_path = os.path.join(base_path, episode_range)
        
        print(f"\n{'='*50}")
        print(f"处理范围: {episode_range}")
        print(f"路径: {current_path}")
        print(f"{'='*50}")
        
        if not os.path.exists(current_path):
            print(f"  跳过: 路径不存在 - {current_path}")
            continue
        
        # 获取并随机排序场景文件
        scene_files = get_scene_files(current_path)
        random.shuffle(scene_files)
        print(f"  共 {len(scene_files)} 个场景，随机处理顺序")
        
        # 处理每个场景
        for scene_name in scene_files:
            args.scene_name = scene_name
            args.img_path = os.path.join(current_path, "images")
            args.instructions_path = os.path.join(current_path, "instructions")
            
            # 创建指令目录（如果不存在）
            os.makedirs(args.instructions_path, exist_ok=True)
            
            process_scene(args)


if __name__ == "__main__":
    main()
