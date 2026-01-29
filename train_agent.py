import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium import spaces
import warnings

# 过滤 gymnasium 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# SB3 核心组件
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# 引入环境加载函数
from dm_rl_env import load_env


class DMControlWrapper(gym.Env):
    def __init__(self, dm_env_instance=None):
        if dm_env_instance is None:
            self.env = load_env()
        else:
            self.env = dm_env_instance

        self.metadata = {'render.modes': ['rgb_array']}
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )
        obs_spec = self.env.observation_spec()
        dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False
        info = {}
        if reward < -5.0: info['collision'] = True
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.physics.render(camera_id=-1, height=480, width=640)


def make_env_fn(rank, seed=0):
    def _init():
        env = DMControlWrapper()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":

    # === 优化并行数 ===
    max_cores = os.cpu_count() or 4
    # IK 计算非常消耗 CPU，为了稳定性，请只使用 60-70% 的核心
    num_cpu = max(1, int(max_cores * 0.85))
    print(f"物理核心: {max_cores}, 正在使用 {num_cpu} 个并行环境...")

    env_fns = [make_env_fn(i, seed=100) for i in range(num_cpu)]

    env = SubprocVecEnv(env_fns, start_method='fork')

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        tensorboard_log="./logs/tb_logs/",
        device="cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path='./logs/',
        name_prefix='aubo_safe_rl'
    )

    print("开始避障增强训练...")
    try:
        model.learn(total_timesteps=5000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("手动停止训练，正在保存...")
    finally:
        # 无论是否出错，最后都保存一次
        model.save("aubo_scan_safe_policy_final")
        env.close()
        print("模型已保存，环境已关闭。")