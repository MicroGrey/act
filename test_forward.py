# simulate_hdf5_actions.py
import h5py
import numpy as np
import torch
from sim_env import make_sim_env  # 你的仿真环境创建函数
from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from utils import sample_box_pose, sample_insertion_pose
import matplotlib.pyplot as plt
import os

# 读取 HDF5 动作
def load_actions(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['/actions'][:]  # shape (400, 14)
    return actions

def run_simulation(actions, task_name='sim_transfer_cube', onscreen_render=True, save_dir='./simulation_videos'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建仿真环境
    env = make_sim_env(task_name)
    max_timesteps = actions.shape[0]
    state_dim = actions.shape[1]

    # 如果是 transfer_cube 或 insertion，需要设置初始 BOX_POSE
    from sim_env import BOX_POSE
    if 'sim_transfer_cube' in task_name:
        BOX_POSE[0] = sample_box_pose()
    elif 'sim_insertion' in task_name:
        BOX_POSE[0] = np.concatenate(sample_insertion_pose())

    ts = env.reset()
    
    qpos_list = []
    target_qpos_list = []
    rewards = []
    image_list = []

    # onscreen 渲染准备
    if onscreen_render:
        plt.ion()
        fig, ax = plt.subplots()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id='angle'))

    for t in range(max_timesteps):
        target_qpos = actions[t]
        ts = env.step(target_qpos)  # 执行动作

        # 收集数据用于可视化
        qpos_list.append(ts.observation['qpos'])
        target_qpos_list.append(target_qpos)
        rewards.append(ts.reward)
        if onscreen_render:
            image = env._physics.render(height=480, width=640, camera_id='angle')
            plt_img.set_data(image)
            plt.pause(DT)
            image_list.append(image)

    plt.close()
    episode_return = np.sum([r for r in rewards if r is not None])
    print(f'Episode return: {episode_return}')

    # 可选：保存视频
    from visualize_episodes import save_videos
    save_videos(image_list, DT, video_path=os.path.join(save_dir, 'hdf5_episode.mp4'))

if __name__ == '__main__':
    hdf5_path = './output/trt_actions.hdf5'
    actions = load_actions(hdf5_path)
    run_simulation(actions)
