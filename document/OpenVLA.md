## 🚀 第一步：环境配置

1. 激活环境

   ```terminal
   conda activate RoboTwin
   cd RoboTwin/policy/openvla-oft
   ```

2. 安装依赖

   ```terminal
   pip install -e .
   pip install packaging ninja
   ninja --version
   pip install "flash-attn==2.5.5" --no-build-isolation
   # 若后续训练遇到 diffusers 报错，可按官方提示回退版本
   # pip install diffusers==0.33.1
   ```

---

## 🔨 第二步：数据预处理

1. 提取并对齐 ALOHA 标准格式（HDF5 $\rightarrow$ ALOHA）

   1. 替换 `preprocess_aloha.sh`

        ```sh
        #!/bin/bash
        TASK=$1
        CONFIG=$2
        
        # 动态拼接路径 (假设你在 policy/openvla-oft 目录下执行)
        DATASET_PATH="../../data/${TASK}/${CONFIG}/data"
        INSTRUCTION_DIR="../../data/${TASK}/${CONFIG}/instructions"
        OUT_BASE_DIR="../../data/${TASK}/${CONFIG}/processed_openvla"
        
        echo "正在处理任务: ${TASK} | 配置: ${CONFIG}"
        echo "数据来源: ${DATASET_PATH}"
        
        python preprocess_aloha.py \
            --dataset_path $DATASET_PATH \
            --out_base_dir $OUT_BASE_DIR \
            --instruction_dir $INSTRUCTION_DIR \
            --percent_val 0.05
        ```

   2. 替换 `preprocess_aloha.py` 中的 `main` 函数

        ```sh
        import re # 记得在文件顶部 import re，或者直接放在这里也行
        
        def main(args):
            input_dir = args.dataset_path
            output_base = args.out_base_dir
            resize_size = args.img_resize_size
            instruction_dir = args.instruction_dir
        
            # 修复1：去掉 [:100]，读取全部采集的数据
            all_eps = sorted(glob(os.path.join(input_dir, "*.hdf5")))
            random.seed(42)
            random.shuffle(all_eps)
        
            n_val = int(len(all_eps) * args.percent_val)
            train_eps = all_eps[:-n_val]
            val_eps = all_eps[-n_val:]
        
            print(f"Total episodes: {len(all_eps)}")
            print(f"Train: {len(train_eps)}, Val: {len(val_eps)}")
        
            for split_name, split_eps in [("train", train_eps), ("val", val_eps)]:
                out_dir = os.path.join(output_base, split_name)
                os.makedirs(out_dir, exist_ok=True)
                for ep in tqdm(split_eps, desc=f"Processing {split_name}"):
                    # 修复2：从当前处理的 hdf5 文件名中提取真实的 episode 编号
                    ep_basename = os.path.basename(ep) # 例如 "episode24.hdf5" 或 "episode_24.hdf5"
                    orig_idx_match = re.search(r'\d+', ep_basename)
                    if not orig_idx_match:
                        continue
                    orig_idx = int(orig_idx_match.group())
        
                    out_path = os.path.join(out_dir, ep_basename)
                    try:
                        # 传入真实的 orig_idx，确保 json 和 hdf5 完美对应！
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
        ```

   3. 终端运行

      ```terminal
      bash preprocess_aloha.sh beat_block_hammer demo_clean
      bash preprocess_aloha.sh beat_block_hammer demo_randomized
      bash preprocess_aloha.sh handover_block demo_clean
      bash preprocess_aloha.sh handover_block demo_randomized
      bash preprocess_aloha.sh click_alarmclock demo_clean
      bash preprocess_aloha.sh move_playingcard_away demo_clean
      bash preprocess_aloha.sh open_laptop demo_clean
      ```

2. 编写专属 TFDS Builder 脚本

   1. 复制 Builder 模板脚本

      ```terminal
      cp datasets/move_can_pot_builder.py datasets/beat_block_hammer_builder.py
      cp datasets/move_can_pot_builder.py datasets/handover_block_builder.py
      cp datasets/move_can_pot_builder.py datasets/click_alarmclock_builder.py
      cp datasets/move_can_pot_builder.py datasets/move_playingcard_away_builder.py
      cp datasets/move_can_pot_builder.py datasets/open_laptop_builder.py
      ```

   2. 修改 `beat_block_hammer_builder.py`（其他任务只需要将下面的`beat_block_hammer`替换为相应的任务名称即可）

      ```python
      # 1.修改类定义
      class aloha_beat_block_hammer(MultiThreadedDatasetBuilder):
          VERSION = tfds.core.Version("1.0.0")
          RELEASE_NOTES = {
              "1.0.0": "Initial release for RoboTwin beat_block_hammer dataset.",
          }
      # 2. 修改数据读取路径
      train_files = glob.glob(
          "../../data/beat_block_hammer/demo_clean/processed_openvla/train/*.hdf5"
      )
      val_files = glob.glob(
          "../../data/beat_block_hammer/demo_clean/processed_openvla/val/*.hdf5"
      )
      # 3. 修改执行入口
      builder = aloha_beat_block_hammer()
      ```

3. 在 OpenVLA 源码中“注册户口”

   1. 注册数据集配置：修改`prismatic/vla/datasets/rlds/oxe/configs.py`

      ```python
      # OXE_DATASET_CONFIGS 字典的末尾添加:
      "aloha_beat_block_hammer": {
          "image_obs_keys": {
              "primary": "image",
              "secondary": "low_cam_image",
              "left_wrist":"left_wrist_image",
              "right_wrist":"right_wrist_image",
          },
          "depth_obs_keys": {
              "primary": None,
              "secondary": None,
              "wrist": None,
          },
          "state_obs_keys": ["state"],  
          "state_encoding": StateEncoding.JOINT_BIMANUAL,  
          "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,  
      },
      "aloha_handover_block": {
          "image_obs_keys": {
              "primary": "image",
              "secondary": "low_cam_image",
              "left_wrist":"left_wrist_image",
              "right_wrist":"right_wrist_image",
          },
          "depth_obs_keys": {
              "primary": None,
              "secondary": None,
              "wrist": None,
          },
          "state_obs_keys": ["state"],  
          "state_encoding": StateEncoding.JOINT_BIMANUAL,  
          "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,  
      },
      "aloha_click_alarmclock": {
          "image_obs_keys": {"primary": "image", "secondary": "low_cam_image", "left_wrist":"left_wrist_image", "right_wrist":"right_wrist_image"},
          "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
          "state_obs_keys": ["state"],  
          "state_encoding": StateEncoding.JOINT_BIMANUAL,  
          "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,  
      },
      "aloha_move_playingcard_away": {
          "image_obs_keys": {"primary": "image", "secondary": "low_cam_image", "left_wrist":"left_wrist_image", "right_wrist":"right_wrist_image"},
          "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
          "state_obs_keys": ["state"],  
          "state_encoding": StateEncoding.JOINT_BIMANUAL,  
          "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,  
      },
      "aloha_open_laptop": {
          "image_obs_keys": {"primary": "image", "secondary": "low_cam_image", "left_wrist":"left_wrist_image", "right_wrist":"right_wrist_image"},
          "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
          "state_obs_keys": ["state"],  
          "state_encoding": StateEncoding.JOINT_BIMANUAL,  
          "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,  
      },
      ```

   2. 注册数据预处理映射：修改`prismatic/vla/datasets/rlds/oxe/transforms.py`

      ```python
      # OXE_STANDARDIZATION_TRANSFORMS 字典末尾添加:
      "aloha_beat_block_hammer": aloha_dataset_transform,
      "aloha_handover_block": aloha_dataset_transform,
      "aloha_click_alarmclock": aloha_dataset_transform,
      "aloha_move_playingcard_away": aloha_dataset_transform,
      "aloha_open_laptop": aloha_dataset_transform,
      ```

   3. 注册数据混合权重：修改`prismatic/vla/datasets/rlds/oxe/mixtures.py`

      ```python
      # OXE_NAMED_MIXTURES 字典末尾添加:
      "aloha_beat_block_hammer": [
          ("aloha_beat_block_hammer", 1.0),
      ],
      "aloha_handover_block": [
          ("aloha_handover_block", 1.0),
      ],
      "aloha_click_alarmclock": [
        	("aloha_click_alarmclock", 1.0),
      ],
      "aloha_move_playingcard_away": [
        	("aloha_move_playingcard_away", 1.0),
      ],
      "aloha_open_laptop": [
        	("aloha_open_laptop", 1.0),
      ],
      ```

4. 正式生成 RLDS 数据流

   ```terminal
   python -m datasets.beat_block_hammer_builder
   python -m datasets.handover_block_builder
   python -m datasets.click_alarmclock_builder
   python -m datasets.move_playingcard_away_builder
   python -m datasets.open_laptop_builder
   
   # 迁移数据到/ssd2下，并建立软链接
   mv ~/tensorflow_datasets /ssd2/cunxuou/
   ln -s /ssd2/cunxuou/tensorflow_datasets ~/tensorflow_datasets
   ls -l ~/tensorflow_datasets
   ```

---

## 🔥 第三步：开启训练

1. 修改`finetune_aloha.sh`

   旧版：

   ```sh
   #!/bin/bash
   
   # 开启离线记录模式，防止 WandB 没登录导致训练崩溃挂起
   export WANDB_MODE=offline
   # 使用 HuggingFace 国内镜像源，防止下载 14GB 的 OpenVLA-7B 骨干网络时断连
   export HF_ENDPOINT=https://hf-mirror.com
   
   # --nproc-per-node 8：火力全开，调用全部 8 张 A800！
   torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
     --vla_path "openvla/openvla-7b" \
     --data_root_dir "/ssd2/cunxuou/tensorflow_datasets/" \
     --dataset_name "aloha_beat_block_hammer" \
     --run_root_dir "/ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints" \
     --use_l1_regression True \
     --use_diffusion False \
     --use_film True \
     --num_images_in_input 3 \
     --grad_accumulation_steps 1 \
     --use_proprio True \
     --batch_size 4 \
     --learning_rate 5e-4 \
     --num_steps_before_decay 10000 \
     --max_steps 20000 \
     --use_val_set True \
     --val_freq 500 \
     --save_freq 2000 \
     --save_latest_checkpoint_only False \
     --image_aug True \
     --lora_rank 32 \
     --wandb_entity "robotwin_local" \
     --wandb_project "openvla_finetune" \
     --run_id_note "beat_block_hammer_8gpu"
   ```

   新版（其他任务只需修改`--dataset_name`和`--run_id_note`即可）：

   ```sh
   #!/bin/bash
   
   export WANDB_MODE=offline
   export HF_ENDPOINT=https://hf-mirror.com
   
   torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
     --vla_path "openvla/openvla-7b" \
     --data_root_dir "/ssd2/cunxuou/tensorflow_datasets/" \
     --dataset_name "aloha_click_alarmclock" \
     --run_root_dir "/ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints" \
     --use_l1_regression True \
     --use_diffusion False \
     --use_film True \
     --num_images_in_input 3 \
     --grad_accumulation_steps 1 \
     --use_proprio True \
     --batch_size 4 \
     --learning_rate 5e-4 \
     --num_steps_before_decay 50000 \
     --max_steps 100005 \
     --use_val_set True \
     --val_freq 1000 \
     --save_freq 5000 \
     --save_latest_checkpoint_only False \
     --image_aug True \
     --lora_rank 32 \
     --wandb_entity "ygtreceplain" \
     --wandb_project "openvla_finetune" \
     --run_id_note "click_alarmclock_8gpu"
   ```

2. 环境修复

   ```terminal
   pip install diffusers==0.28.0 huggingface_hub==0.23.0
   ```

3. 启动微调

   ```terminal
   tmux new -s train_robotwin_openvla
   conda activate RoboTwin
   cd /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft
   export HF_ENDPOINT=https://hf-mirror.com
   
   bash finetune_aloha.sh
   
   tmux attach -t train_robotwin_openvla
   ```

4. 合并权重

   1. 修改 `merge_lora.sh`（检查点替换为相应的检查点）

      ```sh
      python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_open_laptop+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--open_laptop_8gpu--30000_chkpt
      ```

   2. 运行

      ```sh
      bash merge_lora.sh
      ```

   3. 复制说明书（检查点替换为相应的检查点）

      ```terminal
      cp /home/cunxuou/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/*.py /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_open_laptop+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--open_laptop_8gpu--30000_chkpt
      ```

      

---

## 🧪 第四步：模型验证

（检查点替换为相应的检查点）

```terminal
bash eval.sh open_laptop demo_clean /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_open_laptop+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--open_laptop_8gpu--30000_chkpt 0 0 aloha_open_laptop
```

```terminal
tmux new -s test0_robotwin_openvla
conda activate RoboTwin
cd /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft
export HF_ENDPOINT=https://hf-mirror.com

bash eval.sh open_laptop demo_randomized /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_open_laptop+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--open_laptop_8gpu--30000_chkpt 0 0 aloha_open_laptop

tmux attach -t test0_robotwin_openvla
```

```terminal
tmux new -s test1_robotwin_openvla
conda activate RoboTwin
cd /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft
export HF_ENDPOINT=https://hf-mirror.com

bash eval.sh click_alarmclock demo_randomized /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_click_alarmclock+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--click_alarmclock_8gpu--30000_chkpt 0 1 aloha_click_alarmclock

tmux attach -t test1_robotwin_openvla
```

```terminal
tmux new -s test2_robotwin_openvla
conda activate RoboTwin
cd /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft
export HF_ENDPOINT=https://hf-mirror.com

bash eval.sh move_playingcard_away demo_randomized /ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints/openvla-7b+aloha_move_playingcard_away+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--move_playingcard_away_8gpu--30000_chkpt 0 2 aloha_move_playingcard_away

tmux attach -t test2_robotwin_openvla
```
