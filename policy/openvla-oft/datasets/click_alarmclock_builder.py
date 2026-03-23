from typing import Iterator, Tuple, Any
import os
import h5py
import glob
import numpy as np
import tensorflow_datasets as tfds
import random
from datasets.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    '''
    把 HDF5 文件转换成 RLDS 格式样本
    输入是一批 HDF5 文件路径 paths
    输出相应的 RLDS 格式样本
    '''
    print(f"[INFO] Generating examples from {len(paths)} paths") # 打印一共要处理多少个文件
    # 逐个处理每个 episode 的 HDF5 路径
    for path in paths:
        print(f"[INFO] Parsing file: {path}")
        with h5py.File(path, "r") as f:
            # 列出该预处理 HDF5 必须存在的 dataset 路径
            required_keys = [
                "/relative_action",
                "/head_camera_image",
                "/left_wrist_image",
                "/right_wrist_image",
                "/low_cam_image",
                "/action",
                "/seen",
            ]
            # 若缺少任意一个 key，就打印缺了哪些，然后 continue 跳过该文件
            if not all(k in f for k in required_keys):
                for key in required_keys:
                    if key not in f:
                        print(f"[ERROR] Missing key: {key} in {path}")
                print(f"[WARNING] Missing expected keys in {path}, skipping")
                continue
            T = f["/action"].shape[0] # 动作序列长度 T
            actions = f["/action"][1:].astype(np.float32)  # 取 action 从第 2 个时间步到最后，形状 (T-1, 14)。含义：第 i 步要预测的动作，对应“下一时刻的关节状态/动作”
            head = f["/head_camera_image"][ : T-1 ].astype(np.uint8) # 头戴相机图像，形状 (T-1, 256, 256, 3)。观测是 t=0..T-2，动作用的是 action[1:]，对应在观测 t 上预测从当前到下一时刻的动作（或下一时刻目标）
            left = f["/left_wrist_image"][ : T-1].astype(np.uint8) # 左手腕相机图像，形状 (T-1, 256, 256, 3)
            right = f["/right_wrist_image"][ :T-1].astype(np.uint8) # 右手腕相机图像，形状 (T-1, 256, 256, 3)
            low = f["/low_cam_image"][ : T-1].astype(np.uint8) # 低视角相机图像，形状 (T-1, 256, 256, 3)
            states = f["/action"][: T - 1].astype(np.float32)  # (T-1, 14)
            seen = [
                # 读出 /seen 里的所有语言指令字符串；若是 bytes 则解码成 UTF-8
                s.decode("utf-8") if isinstance(s, bytes) else s for s in f["/seen"][()]
            ]
            T -= 1 # 把 T 更新为 有效步数 T-1

            if not seen:
                print(f"[ERROR] No 'seen' instructions found in {path}")
                continue

            if not (
                head.shape[0]
                == left.shape[0]
                == right.shape[0]
                == low.shape[0]
                == T
                == states.shape[0]
            ):
                print(f"[ERROR] Data length mismatch in {path}")
                continue

            instruction = seen # 每个 step 都用同一组 seen 指令列表

            steps = []
            for i in range(T):
                step = {
                    "observation": {
                        "image": head[i],
                        "left_wrist_image": left[i],
                        "right_wrist_image": right[i],
                        "low_cam_image": low[i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == T - 1 else 0.0),
                    "is_first": np.bool_(i == 0),
                    "is_last": np.bool_(i == T - 1),
                    "is_terminal": np.bool_(i == T - 1),
                    "language_instruction": instruction,
                }
                steps.append(step)

            print(f"[INFO] Yielding {len(steps)} steps from {path}")
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


class aloha_click_alarmclock(MultiThreadedDatasetBuilder):
    '''
    把预处理后的 processed_openvla/train/*.hdf5 和 val/*.hdf5 按 TFDS(TensorFlow Datasets)的格式描述成一个“数据集”，用于训练和验证
    '''
    VERSION = tfds.core.Version("1.0.0") # 数据集版本号 1.0.0。TFDS 会用它管理缓存/下载/重建。
    RELEASE_NOTES = { 
        "1.0.0": "Initial release for RoboTwin click_alarmclock dataset.", # 版本发布说明
    }

    N_WORKERS = 1 # 并行处理线程数，这里设为 1 是因为每个文件处理时间不长，不需要太多并行
    MAX_PATHS_IN_MEMORY = 100 # 内存中缓存的最大文件数，这里设为 100 是因为每个文件处理时间不长，不需要太多内存缓存
    PARSE_FCN = _generate_examples # 解析函数，把 HDF5 文件转换成 RLDS 格式样本

    def _info(self) -> tfds.core.DatasetInfo:
        '''
        数据结构约定，决定了训练代码读数据时“期望看到什么字段、字段形状是什么”
        '''
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "low_cam_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,), dtype=np.float32
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32), 
                            "reward": tfds.features.Scalar(dtype=np.float32), 
                            "is_first": tfds.features.Scalar(dtype=np.bool_), # 是否是第一个 step
                            "is_last": tfds.features.Scalar(dtype=np.bool_), # 是否是最后一个 step
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_), # 是否是终止 step
                            "language_instruction": tfds.features.Sequence(
                                tfds.features.Text()
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        '''
        数据集划分，返回训练集和验证集的文件列表
        '''
        train_files = glob.glob(
            "../../data/click_alarmclock/demo_clean/processed_openvla/train/*.hdf5"
        )
        val_files = glob.glob(
            "../../data/click_alarmclock/demo_clean/processed_openvla/val/*.hdf5"
        )

        print(f"[INFO] Found {len(train_files)} training files")
        print(f"[INFO] Found {len(val_files)} validation files")

        return {
            "train": train_files,
            "val": val_files,
        }


if __name__ == "__main__":
    builder = aloha_click_alarmclock()
    builder.download_and_prepare()
