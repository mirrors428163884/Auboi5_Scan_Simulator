import numpy as np
import os
import warnings
import gymnasium as gym
from gymnasium import spaces

# 过滤 gymnasium 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# SB3 核心组件
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# --- 修改这里: 引入带场景的新环境 ---
from dm_rl_envwithscene import load_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


class DMControlWrapper(gym.Env):
    def __init__(self, dm_env_instance=None):
        if dm_env_instance is None:
            self.env = load_env()
        else:
            self.env = dm_env_instance

        self.metadata = {'render.modes': ['rgb_array']}

        # 获取动作空间
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )

        # 获取观测空间 (自动展平所有 dict 观测)
        obs_spec = self.env.observation_spec()
        dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 注意: dm_control 的 reset 返回 TimeStep
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0

        # dm_control 的 last() 表示结束，但我们需要区分是成功结束还是碰撞结束
        terminated = time_step.last()
        truncated = False

        info = {}
        # 如果奖励很低，说明发生了碰撞，可以在这里记录 info
        if reward < -10.0: info['collision'] = True

        return obs, reward, terminated, truncated, info

    def render(self):
        # 渲染相机 ID -1 (自由视角) 或 指定相机名
        return self.env.physics.render(camera_id=-1, height=480, width=640)


def make_env_fn(rank, seed=0):
    def _init():
        env = DMControlWrapper()
        env = Monitor(env)  # 监控数据
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":

    # === 路径设置 ===
    log_dir = "./logs/scene_rl/"
    os.makedirs(log_dir, exist_ok=True)

    # === 优化并行数 ===
    max_cores = os.cpu_count() or 4
    # IK 计算+物理仿真非常消耗 CPU，建议留有余量
    num_cpu = max(1, int(max_cores * 0.85))
    print(f"检测到物理核心: {max_cores}, 将使用 {num_cpu} 个并行环境进行训练...")

    # 创建并行环境
    env_fns = [make_env_fn(i, seed=200) for i in range(num_cpu)]
    env = SubprocVecEnv(env_fns, start_method='spawn')  # 'spawn' 在某些系统上更稳定

    # 自动计算观测值和奖励的移动平均值和方差,从而将输入缩放到 [-1, 1] 甚至更标准的正态分布，极大加速收敛和提高精度
    env = VecNormalize(env,
                       norm_obs=True,
                       norm_reward=True,
                       clip_obs=10.,
                       gamma=0.99)

    # === PPO 模型配置 ===
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # 增加步数，因为场景交互需要更多步骤
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # 熵系数，鼓励探索
        tensorboard_log=log_dir,
        device="cpu"  # 只有几维向量输入，CPU 通常比 GPU 快 (无图像输入)
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_cpu,
        save_path=log_dir,
        name_prefix='aubo_scene_scan'
    )

    print("开始场景扫描强化训练...")
    print(f"目标: 探测器末端需对准样品 {load_env().task.path_points[0][:3]} 附近...")

    try:
        model.learn(total_timesteps=20000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("手动停止训练，正在保存...")
    finally:
        model.save("aubo_scene_scan_final")
        env.close()
        print("模型已保存: aubo_scene_scan_final.zip")