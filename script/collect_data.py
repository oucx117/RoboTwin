import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    '''
    动态导入任务环境类，返回任务环境实例
    '''
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):
    '''
    主函数，负责初始化任务环境、加载任务配置、运行任务
    '''
    # 1. 动态导入任务环境类，返回任务环境实例
    task = class_decorator(task_name)

    # 2. 加载任务配置，并把全部内容解析成一个字典 args
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 3. 设置任务名称
    args['task_name'] = task_name

    # 4. 加载机器人模型配置
    embodiment_type = args.get("embodiment")
    
    # 5. 加载机器人模型配置文件，并把全部内容解析成一个字典 _embodiment_types
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        '''
        根据机器人模型类型获取机器人模型文件路径
        '''
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    # 6. 根据机器人模型类型设置机器人模型文件路径
    if len(embodiment_type) == 1:
        # 同构双臂
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        # 异构双臂
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    # 7. 根据机器人模型文件路径获取每只臂的详细关节配置
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # 8. 设置机器人模型名称
    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    '''
    运行任务，收集种子（阶段一）和数据（阶段二）
    '''
    # epid：当前尝试的 seed 编号（每次 +1，不管成功失败）
    # suc_num：成功收集的种子数量（每次成功后 +1）
    # fail_num：失败收集的种子数量（每次失败后 +1）
    # seed_list：成功收集的种子列表
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        # 断点续传
        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        # 循环收集种子，直到达到目标数量
        while suc_num < args["episode_num"]:
            # 尝试收集种子，如果成功则保存种子，否则记录失败
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args) # 用当前 seed 初始化场景
                TASK_ENV.play_once() # 执行专家动作脚本

                # 检查是否成功收集到种子：（规划成功 且 执行成功）
                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num) # 保存关节轨迹数据
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env() # 关闭场景，清理环境
                # 如果需要渲染，则关闭渲染窗口（阶段二才需要渲染）
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env() # 关闭场景，清理环境

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env() # 关闭场景，清理环境

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            # 每次循环都覆盖一次 seed.txt 文件，为了实现断点续传
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        # 跳过整个阶段一，直接读取之前保存好的 seed 列表，供阶段二使用
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False # 关闭运动规划
        args["render_freq"] = 0 # 关闭渲染
        args["save_data"] = True # 开启数据保存

        clear_cache_freq = args["clear_cache_freq"] # 每隔多少步清除一次缓存

        st_idx = 0 # 当前采集的 episode 索引

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        # 断点续传
        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            # 用当前 seed 初始化场景
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            # 读取阶段一保存的关节轨迹数据
            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"] # 设置左臂关节轨迹
            args["right_joint_path"] = traj_data["right_joint_path"] # 设置右臂关节轨迹
            # 告诉环境把这两条路径设置成后续回放要走的轨迹
            # 这样 play_once() 在 need_plan=False 时就会按路径执行并采数据
            TASK_ENV.set_path_lst(args) 

            # scene_info.json 用来保存每个 episode 的元信息，比如选了哪只手臂、物体哪个外观 id 等，对应 play_once 函数的返回值
            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            # 回放轨迹并采集数据
            info = TASK_ENV.play_once()
            # 把本 episode 的 info 存入 info_db
            info_db[f"episode_{episode_idx}"] = info
            # 写回 scene_info.jso
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video() # 合并 pkl 文件为 hdf5 文件
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        # 生成语言指令
        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
