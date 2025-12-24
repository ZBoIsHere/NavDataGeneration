import os
import sys
import time
import gzip
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

import google.generativeai as genai
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, InternalServerError

from gemini_annotation_prompt import (
    viewpoint_template,
    direction_template,
    remaining_prompt
)


API_KEYS = [
    'AIzaSyCViEpTPJPKknKMf_y-xkyiErk7qI1603Y',
]


def load_instructions(file_path: str) -> Dict:
    """加载已有的指令数据"""
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"加载指令文件失败 {file_path}: {str(e)}")


def save_instructions(data: Dict, file_path: str, debug: bool = False) -> None:
    """保存指令数据到gzip压缩文件（和可选的json文件）"""
    try:
        json_str = json.dumps(data, indent=4)
        
        # 保存gzip文件
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            f.write(json_str)
        
        # 如果debug模式，也保存普通json文件
        if debug:
            json_path = file_path.replace('.gz', '')
            with open(json_path, 'w', encoding='utf-8') as f:
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


def load_image(image_path: str) -> Image.Image:
    """加载图片文件为PIL Image对象"""
    try:
        return Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"加载图片失败 {image_path}: {str(e)}")


def build_prompt_gemini(episode: Dict, img_dir: str) -> List[Any]:
    """构建发送给Gemini模型的提示（混合文本和图片对象）"""
    episode_id = episode["episode_id"]
    img_path = os.path.join(img_dir, episode_id)
    img_files = sorted(
        [f for f in os.listdir(img_path) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0])
    )
    
    prompt = []
    
    # 添加专家角色描述
    prompt.append(
        "You are an expert in object navigation and vision-language navigation. "
        "You have been provided with a sequence of viewpoints along a navigation path, "
        "represented by images and corresponding action descriptions. "
        "Your task is to generate clear and concise navigation instructions based on these images and actions. "
        "The format of the instructions should be similar to those found in R2R and RxR datasets."
    )
    
    prompt.append("Here are the viewpoints and actions:")
    
    # 添加每个视角
    for i, img_file in enumerate(img_files):
        # 添加视角文本
        prompt.append(viewpoint_template.format(p=i))
        
        # 添加图片
        full_img_path = os.path.join(img_path, img_file)
        try:
            pil_image = load_image(full_img_path)
            prompt.append(pil_image)
        except Exception as e:
            raise RuntimeError(f"处理图片失败 {full_img_path}: {str(e)}")
        
        # 添加动作描述（如果不是最后一张）
        if i + 1 < len(img_files) and i < len(episode["actions"]):
            action_text = episode["actions"][i]
            prompt.append(direction_template.format(action=action_text, p=i+1))
    
    # 添加最后的总结指令
    prompt.append(remaining_prompt.format(view_num=len(img_files)))
    
    return prompt


def call_gemini_api(prompt: List[Any], episode_id: str, model: genai.GenerativeModel) -> str:
    """调用Gemini API并处理响应"""
    try:
        # 为避免速率限制，添加短暂延迟
        time.sleep(1.5)
        
        response = model.generate_content(
            prompt,
            request_options={"timeout": 10}
        )
    except ResourceExhausted as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
            raise RuntimeError(f"配额超限 [{episode_id}]: {error_msg}")
        else:
            raise RuntimeError(f"资源耗尽 [{episode_id}]: {error_msg}")
    except InvalidArgument as e:
        error_msg = str(e)
        raise RuntimeError(f"无效参数 [{episode_id}]: {error_msg}")
    except InternalServerError as e:
        error_msg = str(e)
        raise RuntimeError(f"服务器错误 [{episode_id}]: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            raise RuntimeError(f"速率限制 [{episode_id}]: {error_msg}")
        elif "content filtering policy" in error_msg.lower() or "inappropriate content" in error_msg.lower():
            raise RuntimeError(f"内容违规 [{episode_id}]: {error_msg}")
        else:
            raise RuntimeError(f"API错误 [{episode_id}]: {error_msg}")
    
    # 检查响应内容
    if not hasattr(response, 'text') or not response.text:
        raise RuntimeError(f"空响应或无效响应 [{episode_id}]")
    
    # 提取文本内容
    content = response.text.strip()
    if not content:
        raise RuntimeError(f"空指令文本 [{episode_id}]")
    
    return content.lstrip("Instruction: ")


def process_episode(episode: Dict, img_dir: str, episode_id: str, model: genai.GenerativeModel) -> Optional[str]:
    """处理单个episode，生成指令"""
    start_time = time.time()
    print(f"  处理 episode: {episode_id}")
    
    # 构建提示
    try:
        prompt = build_prompt_gemini(episode, img_dir)
    except Exception as e:
        print(f"  构建提示失败: {str(e)}")
        return None
    
    # 调用API
    try:
        instruction = call_gemini_api(prompt, episode_id, model)
        print(f"  生成指令: {instruction}")
        return instruction
    except Exception as e:
        print(f"  生成指令失败: {str(e)}")
        return None
    finally:
        end_time = time.time()
        print(f"  处理耗时: {end_time - start_time:.2f} 秒")


def process_scene(args: argparse.Namespace, model: genai.GenerativeModel) -> None:
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
        instruction = process_episode(episode, img_dir, episode_id, model)
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
                save_instructions(output_data, instructions_file, args.debug)
                print(f"  已保存进度，处理了 {processed_count} 个episodes")
                save_counter = 0
    
    # 最终保存
    save_instructions(output_data, instructions_file, args.debug)
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


def initialize_gemini_model() -> genai.GenerativeModel:
    """初始化Gemini模型，随机选择API key"""
    random_key = random.choice(API_KEYS)
    genai.configure(api_key=random_key)
    return genai.GenerativeModel(model_name="gemini-2.5-flash-001")


def main():
    """主函数：参数解析和流程控制"""
    parser = argparse.ArgumentParser(description='VLN指令生成器 (Gemini版本)')
    parser.add_argument("--vln_topo_path", type=str,
                       default="data/task_datasets/vln/hm3d_v2_l3mvn_refine_v2/train/content/episode_num_3100-3199",
                       help="VLN拓扑数据路径")
    parser.add_argument('--episode_range', type=str, default='3100-3499',
                       help="episode范围 (格式: start-end)")
    parser.add_argument("--scene_type", type=str, default="hm3d_v1",
                       help="场景类型")
    parser.add_argument("--split", type=str, default="train",
                       help="数据集划分")
    parser.add_argument("--debug", default="True", help="启用debug模式，同时保存JSON文件")
    args = parser.parse_args()
    
    # 初始化Gemini模型
    try:
        model = initialize_gemini_model()
        print("成功初始化Gemini模型")
    except Exception as e:
        print(f"初始化Gemini模型失败: {str(e)}")
        sys.exit(1)
    
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
            args.instructions_path = os.path.join(current_path, "instructions_gemini")
            
            # 创建指令目录（如果不存在）
            os.makedirs(args.instructions_path, exist_ok=True)
            
            process_scene(args, model)


if __name__ == "__main__":
    main()
