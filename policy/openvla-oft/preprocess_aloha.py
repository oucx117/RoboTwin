import json
import os
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
from glob import glob
import random
from tqdm import tqdm
import re

def decode_and_resize_images(image_bytes_array, size=256):
    '''
    解码和调整图像尺寸
    image_bytes_array: 图像字节流数组
    size: 图像要调整到的尺寸
    '''
    resized = [] # 存储调整后的图像
    for img_bytes in image_bytes_array:
        img_pil = Image.open(BytesIO(img_bytes.tobytes())).convert("RGB") # 解码为 RGB 图像
        img_resized = img_pil.resize((size, size), resample=Image.BICUBIC) # 调整尺寸
        resized.append(np.array(img_resized)) # 转换为 numpy 数组并添加到列表中
    return np.stack(resized) # 将列表中的 numpy 数组堆叠成一个 numpy 数组，形状为 (N, size, size, 3)

def load_instruction(instruction_dir, episode_idx):
    json_path = os.path.join(instruction_dir, f"episode_{episode_idx}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Instruction file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    candidates = data.get("seen", []) + data.get("unseen", [])
    if not candidates:
        raise ValueError(f"No instructions found in {json_path}")
    return random.choice(candidates)

def process_one_episode(input_path, output_path, episode_idx, instruction_dir, resize_size=256):
    '''
    处理单个 episode 的数据：统一四路相机到 256x256, 计算动作的相对差分, 并把对应 episode 的 seen/unseen 语言指令一起打包进新的 hdf5
    input_path: 输入的 hdf5 文件路径
    output_path: 输出的 hdf5 文件路径
    episode_idx: 当前处理的 episode 编号
    instruction_dir: 语言指令所在目录
    resize_size: 图像要调整到的尺寸
    '''
    # 读取输入的 hdf5 文件
    with h5py.File(input_path, "r") as f:
        # 读取原始动作，并转换成 numpy 数组
        action = f["joint_action/vector"][()]
        # 计算相对动作
        rel_action = np.zeros_like(action) # 初始化相对动作
        rel_action[:-1] = action[1:] - action[:-1] # 计算相对动作
        rel_action[-1] = rel_action[-2] # 最后一个相对动作等于倒数第二个相对动作

        # 解码和调整图像尺寸
        head = decode_and_resize_images(f["observation/head_camera/rgb"][()], size=resize_size)
        left = decode_and_resize_images(f["observation/left_camera/rgb"][()], size=resize_size)
        right = decode_and_resize_images(f["observation/right_camera/rgb"][()], size=resize_size)
        front = decode_and_resize_images(f["observation/front_camera/rgb"][()], size=resize_size)

    # 读取该 episode 对应的文本指令
    json_path = os.path.join(instruction_dir, f"episode{episode_idx}.json")
    with open(json_path, "r") as f:
        inst_data = json.load(f)
    seen_list = inst_data.get("seen", [])
    unseen_list = inst_data.get("unseen", [])

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 

    # 创建输出 hdf5 文件
    with h5py.File(output_path, "w") as f:
        f.create_dataset("head_camera_image", data=head, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("left_wrist_image", data=left, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("right_wrist_image", data=right, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("low_cam_image", data=front, dtype="uint8", chunks=(1, resize_size, resize_size, 3)) 
        f.create_dataset("action", data=action)
        f.create_dataset("relative_action", data=rel_action)
        f.create_dataset("seen", data=np.array(seen_list, dtype=h5py.string_dtype(encoding="utf-8")))
        f.create_dataset("unseen", data=np.array(unseen_list, dtype=h5py.string_dtype(encoding="utf-8")))


def main(args):
    '''
    主函数，处理数据集
    '''
    input_dir = args.dataset_path # 数据集路径（*.hdf5）
    output_base = args.out_base_dir # 输出根目录
    resize_size = args.img_resize_size # 图像要调整到的尺寸
    instruction_dir = args.instruction_dir # 语言指令所在目录

    # 读取所有采集的数据
    all_eps = sorted(glob(os.path.join(input_dir, "*.hdf5")))
    random.seed(42)
    random.shuffle(all_eps)

    # 划分训练集和验证集
    n_val = int(len(all_eps) * args.percent_val)
    train_eps = all_eps[:-n_val]
    val_eps = all_eps[-n_val:]

    print(f"Total episodes: {len(all_eps)}")
    print(f"Train: {len(train_eps)}, Val: {len(val_eps)}")

    # 处理训练集和验证集
    for split_name, split_eps in [("train", train_eps), ("val", val_eps)]:
        # 创建输出目录
        out_dir = os.path.join(output_base, split_name)
        os.makedirs(out_dir, exist_ok=True)
        # 处理每个 episode
        for ep in tqdm(split_eps, desc=f"Processing {split_name}"):
            ep_basename = os.path.basename(ep) # 只取文件名（去掉路径），例如 "episode24.hdf5"
            orig_idx_match = re.search(r'\d+', ep_basename) # 从文件名中提取真实的 episode 编号，例如 "24"
            if not orig_idx_match:
                continue
            orig_idx = int(orig_idx_match.group()) # 转换为整数

            out_path = os.path.join(out_dir, ep_basename) # 输出路径，例如 "train/episode24.hdf5"
            try:
                # 处理单个 episode（传入真实的 orig_idx，确保 json 和 hdf5 完美对应）
                process_one_episode(ep, out_path, orig_idx, instruction_dir, resize_size=resize_size)
            except Exception as e:
                print(f"[ERROR] Failed to process {ep}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_base_dir", type=str, required=True)
    parser.add_argument("--instruction_dir", type=str, required=True)
    parser.add_argument("--percent_val", type=float, default=0.05)
    parser.add_argument("--img_resize_size", type=int, default=256)
    args = parser.parse_args()
    main(args)


"""
python preprocess_aloha.py   --dataset_path /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/demo_randomized/data   --out_base_dir /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/processed_openvla/   --percent_val 0.05 --instruction_dir /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/demo_randomized/instructions
"""