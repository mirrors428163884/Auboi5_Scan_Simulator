import time
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation as R

# 必须引用带场景的新环境
try:
    from dm_rl_envwithscene import load_env
except ImportError:
    print("❌ 错误：找不到 dm_rl_envwithscene.py。")
    exit()

# 探测器偏移量
TCP_OFFSET = np.array([0.0, 0.067, 0.0965])

# 尝试导入 OpenCV
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装 opencv-python，无法显示画面，只能打印 log。")


# --- 鼠标交互控制器 ---
class CameraController:
    def __init__(self, physics):
        self.physics = physics
        self.dragging = False
        self.zooming = False
        self.last_x = 0
        self.last_y = 0
        # 初始默认视角
        self.azimuth = 140.0
        self.elevation = -25.0
        self.update_camera()

    def update_camera(self):
        self.physics.model.vis.global_.azimuth = self.azimuth
        self.physics.model.vis.global_.elevation = self.elevation

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x, self.last_y = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zooming = True
            self.last_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_x
                dy = y - self.last_y
                self.azimuth -= dx * 0.5
                self.elevation += dy * 0.5
                self.elevation = np.clip(self.elevation, -89, 89)
                self.last_x, self.last_y = x, y
                self.update_camera()
            elif self.zooming:  # 简单的缩放模拟
                # MuJoCo Python绑定改 distance 比较麻烦，这里暂只处理旋转
                # 某些版本可用 self.physics.model.vis.map.zfar 等参数微调
                pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONUP:
            self.zooming = False


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
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False

        info = {}
        # 判定碰撞或违规 (基于 Reward)
        if reward < -5.0:
            info['collision'] = True

        # --- 详细计算逻辑 (保留不删) ---
        try:
            physics = self.env.physics
            # 1. 获取 Wrist
            if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
                wrist_pos = physics.named.data.xpos['wrist3_Link']
                wrist_mat = physics.named.data.xmat['wrist3_Link'].reshape(3, 3)
            else:
                idx = 6 if physics.data.xpos.shape[0] > 6 else -1
                wrist_pos = physics.data.xpos[idx]
                wrist_mat = physics.data.xmat[idx].reshape(3, 3)

            # 2. 计算 Detector Tip
            tip_pos = wrist_pos + wrist_mat @ TCP_OFFSET
            info['wrist_pos'] = wrist_pos
            info['tip_pos'] = tip_pos

            # 3. 诊断碰撞原因 (Height vs Contact)
            # 如果 reward 很低，检查是否是因为高度太低
            if reward < -5.0:
                if tip_pos[2] < 0.05:  # 对应环境里的 MIN_HEIGHT_LIMIT
                    info['fail_reason'] = "Height Low"
                else:
                    info['fail_reason'] = "Hit Object"

            # 4. 误差计算
            task = self.env.task
            target_pos = getattr(task, '_current_base_target', None)

            if target_pos is not None:
                t_pos = target_pos[:3]
                t_euler = target_pos[3:]

                # 距离
                dist = np.linalg.norm(t_pos - tip_pos)
                info['dist_error'] = dist

                # 角度
                target_rot = R.from_euler('xyz', t_euler, degrees=False)
                target_mat = target_rot.as_matrix()
                r_diff = np.dot(wrist_mat, target_mat.T)
                trace = np.trace(r_diff)
                angle_deg = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))
                info['ang_error'] = angle_deg
            else:
                info['dist_error'] = 0.0
                info['ang_error'] = 0.0

        except:
            info['dist_error'] = -1.0

        return obs, reward, terminated, truncated, info

    def render(self):
        # 使用安全分辨率
        return self.env.physics.render(camera_id=-1, height=480, width=640)


def main():
    model_paths = ["aubo_scene_scan_final.zip", "aubo_scan_safe_policy_final.zip"]
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("❌ 未找到模型文件")
        return

    print(f"✅ 加载模型: {model_path}")

    dm_env = load_env()
    env = DMControlWrapper(dm_env)

    # 鼠标控制器
    cam_ctrl = CameraController(dm_env.physics)

    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    obs, _ = env.reset()

    window_name = "Interactive RL Test"
    if HAS_CV2:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, cam_ctrl.mouse_callback)

    print("\n" + "=" * 80)
    print("🎮 交互式测试启动")
    print("   [鼠标左键拖动] 旋转视角")
    print("   [注意] 如果提示 'Height Low'，说明探测器离桌子太近触发了安全停止")
    print("=" * 80 + "\n")

    step_count = 0
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # --- 控制台打印 (保留您需要的功能) ---
        if step_count % 10 == 0:
            dist_err = info.get('dist_error', 0.0)
            ang_err = info.get('ang_error', 0.0)
            tip_z = info.get('tip_pos', [0, 0, 0])[2]

            # 状态判定
            status_str = "🟢 正常"
            if info.get('collision'):
                reason = info.get('fail_reason', 'Unknown')
                status_str = f"🔴 警告: {reason}"

            if dist_err != -1.0:
                print(f"Step:{step_count:04d} | R:{reward:6.1f} | "
                      f"Err:{dist_err * 1000:5.1f}mm | Ang:{ang_err:5.1f}° | "
                      f"TipZ:{tip_z:.3f}m | {status_str}")

        # --- 画面渲染 ---
        if HAS_CV2:
            rgb_array = env.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            # 绘制数据
            dist_err = info.get('dist_error', 0.0)
            if dist_err != -1.0:
                cv2.putText(bgr_array, f"Err: {dist_err * 1000:.1f} mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(bgr_array, f"Ang: {info.get('ang_error', 0):.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 绘制警告
            if info.get('collision'):
                reason = info.get('fail_reason', 'COLLISION')
                color = (0, 0, 255)
                cv2.putText(bgr_array, reason.upper(), (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow(window_name, bgr_array)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        if terminated or truncated:
            print(f"--- 回合结束 R={total_reward:.1f} ---\n")
            obs, _ = env.reset()
            cam_ctrl.update_camera()  # 保持视角
            step_count = 0
            total_reward = 0

    if HAS_CV2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()