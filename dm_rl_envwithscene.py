import numpy as np
import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from scipy.spatial.transform import Rotation as R

print(">>> Safe-IK Environment (Scene Compatible + Full Logic) Loaded <<<")

try:
    from curve_utils import CurvePathPlanner
    from casadi_ik import Kinematics
except ImportError:
    print("警告: 未找到 curve_utils 或 casadi_ik，请确保文件在目录下。")
    pass

# 常量定义
CONTROL_TIMESTEP = 0.02  # 50Hz
PHYSICS_TIMESTEP = 0.002  # 500Hz

# --- 配置参数 ---
# 保持您原有的配置
MIN_HEIGHT_LIMIT = 0.2
COLLISION_PENALTY = -500.0
NAN_PENALTY = -1000.0
COLLISION_TERMINATE = True

# --- 场景扫描专用参数 ---
TCP_OFFSET = np.array([0.0, 0.067, 0.0965])
SAMPLE_CENTER = np.array([0.0, -0.54, 0.2])
TARGET_SCAN_HEIGHT = 0.08


class AuboSceneScanTask(base.Task):
    def __init__(self, scene_xml_path, robot_xml_path, random_state=None):
        super().__init__(random=random_state)
        self._random_state = self._random

        self.curve_manager = CurvePathPlanner() if 'CurvePathPlanner' in globals() else None

        # IK 只加载纯机器人 XML
        print(f"Loading IK Model from: {robot_xml_path}")
        self.ik_solver = Kinematics("wrist3_Link")
        self.ik_solver.buildFromMJCF(robot_xml_path)

        self.action_scale = 0.005
        self._latency_buffer = collections.deque(maxlen=5)
        self._current_base_target = None

        self.collision_count = 0
        self.ik_fail_count = 0

        # --- 修复 IndexError: 必须在 init 时生成一条路径 ---
        # 即使没有 physics 无法做有效性过滤，也要先生成一条 raw path
        # 这样 train_agent.py 里的 print(path[0]) 就不会报错了
        self.path_points = self._generate_scan_path(SAMPLE_CENTER)
        self.path_index = 0

    def _pose_to_matrix(self, pose_6d):
        t = pose_6d[:3]
        r_euler = pose_6d[3:]
        mat = np.eye(4)
        mat[:3, 3] = t
        r = R.from_euler('xyz', r_euler, degrees=False)
        mat[:3, :3] = r.as_matrix()
        return mat

    # 获取探测器真实状态
    def _get_detector_tip_state(self, physics):
        if 'wrist3_Link' in physics.named.data.xpos.axes.row.names:
            wrist_pos = physics.named.data.xpos['wrist3_Link']
            wrist_mat = physics.named.data.xmat['wrist3_Link'].reshape(3, 3)
        else:
            # 索引回退机制 (有场景时索引通常是6)
            idx = 6 if physics.data.xpos.shape[0] > 6 else -1
            wrist_pos = physics.data.xpos[idx]
            wrist_mat = physics.data.xmat[idx].reshape(3, 3)

        tip_pos = wrist_pos + wrist_mat @ TCP_OFFSET
        return tip_pos, wrist_mat

    # 探测器目标 -> 手腕目标
    def _detector_target_to_wrist_matrix(self, detector_pose_6d):
        target_pos = detector_pose_6d[:3]
        target_euler = detector_pose_6d[3:]
        r = R.from_euler('xyz', target_euler, degrees=False)
        rot_mat = r.as_matrix()
        wrist_pos = target_pos - rot_mat @ TCP_OFFSET
        wrist_mat = np.eye(4)
        wrist_mat[:3, :3] = rot_mat
        wrist_mat[:3, 3] = wrist_pos
        return wrist_mat

    def _safe_ik(self, target_wrist_matrix, q_guess):
        if np.isnan(target_wrist_matrix).any() or np.isnan(q_guess).any():
            return None, "Input contains NaN"

        # 确保 guess 只有 6 维
        if len(q_guess) > 6: q_guess = q_guess[:6]

        try:
            result = self.ik_solver.ik(target_wrist_matrix, q_guess)
            if isinstance(result, tuple):
                q_sol = result[0]
                info = result[1] if len(result) > 1 else "success"
            else:
                q_sol = result
                info = "success"
            return q_sol, info
        except Exception as e:
            self.ik_fail_count += 1
            return None, str(e)

    def _check_collision_and_height(self, physics):
        # 切片检查前6个关节
        if np.isnan(physics.data.qpos[:6]).any() or np.isnan(physics.data.qvel[:6]).any():
            return False, "physics_nan"

        try:
            tip_pos, _ = self._get_detector_tip_state(physics)
            if tip_pos[2] < MIN_HEIGHT_LIMIT:
                return False, "height_violation"
        except:
            pass

        # 智能碰撞过滤 (解决 -500 分问题)
        try:
            if physics.data.ncon > 0:
                for i in range(physics.data.ncon):
                    contact = physics.data.contact[i]
                    id1 = physics.model.geom_bodyid[contact.geom1]
                    id2 = physics.model.geom_bodyid[contact.geom2]

                    name1 = physics.model.id2name(id1, 'body') or "world"
                    name2 = physics.model.id2name(id2, 'body') or "world"

                    # 只有当至少有一个是机器人部件时，才算有效碰撞
                    is_robot_1 = "Link" in name1 or "connector" in name1 or "detector" in name1
                    is_robot_2 = "Link" in name2 or "connector" in name2 or "detector" in name2

                    if is_robot_1 or is_robot_2:
                        return False, f"physical_collision: {name1} <-> {name2}"
        except Exception:
            return False, "collision_check_error"

        return True, "valid"

    # 保留您原有的舒适区检查
    def _is_joint_in_comfort_zone(self, qpos):
        limit_threshold = 2.8
        if np.any(np.abs(qpos) > limit_threshold):
            return False
        return True

    def initialize_episode(self, physics):
        self._randomize_physics(physics)
        self.collision_count = 0

        # 保留原有的路径筛选逻辑 (Try 10 times)
        valid_path = []
        for _ in range(10):
            raw_path = self._generate_scan_path(SAMPLE_CENTER)
            valid_path = self._filter_valid_path(physics, raw_path)
            if len(valid_path) > 20:
                break

        if len(valid_path) == 0:
            # 默认悬停位置
            default_pose = np.concatenate([SAMPLE_CENTER + [0, 0, 0.2], [3.14, 0, 0]])
            valid_path = [default_pose] * 50

        self.path_points = valid_path
        self.path_index = 0

        start_pose_6d = self.path_points[0]
        start_wrist_matrix = self._detector_target_to_wrist_matrix(start_pose_6d)

        q_guess = [0, 0, 0, 0, 0, 0]
        q_start, _ = self._safe_ik(start_wrist_matrix, q_guess)

        if q_start is None:
            q_start = [0, 0, 0, 0, 0, 0]

        # 切片赋值，修复 13-DoF 维度错误
        physics.data.qpos[:6] = q_start
        physics.data.qvel[:6] = 0
        physics.forward()

        self._latency_buffer.clear()
        for _ in range(self._latency_buffer.maxlen):
            self._latency_buffer.append(q_start)

    # 保留并适配您原有的路径过滤逻辑
    def _filter_valid_path(self, physics, raw_path):
        valid_points = []
        qpos_backup = physics.data.qpos.copy()
        last_q = physics.data.qpos[:6].copy()  # 注意切片

        for pose_6d in raw_path:
            # 探测器目标转 wrist 目标
            mat = self._detector_target_to_wrist_matrix(pose_6d)
            q_sol, _ = self._safe_ik(mat, last_q)

            if q_sol is not None:
                if not self._is_joint_in_comfort_zone(q_sol):
                    break

                # 模拟设置状态并检查 (切片)
                physics.data.qpos[:6] = q_sol
                physics.forward()
                is_valid, reason = self._check_collision_and_height(physics)

                if is_valid:
                    valid_points.append(pose_6d)
                    last_q = q_sol
                else:
                    break
            else:
                break

        # 恢复状态
        physics.data.qpos[:] = qpos_backup
        physics.forward()
        return valid_points

    def before_step(self, action, physics):
        if np.isnan(physics.data.qpos[:6]).any():
            return

        if self.path_index < len(self.path_points):
            base_pose_target_6d = self.path_points[self.path_index]
            self.path_index += 1
        else:
            base_pose_target_6d = self.path_points[-1]

        self._current_base_target = base_pose_target_6d

        q_guess = physics.data.qpos[:6].copy()

        residual_pos = action[:3] * self.action_scale
        target_pose_with_residual = base_pose_target_6d.copy()
        target_pose_with_residual[:3] += residual_pos

        target_matrix = self._detector_target_to_wrist_matrix(target_pose_with_residual)
        q_target, _ = self._safe_ik(target_matrix, q_guess)

        if q_target is None:
            base_matrix = self._detector_target_to_wrist_matrix(base_pose_target_6d)
            q_target, _ = self._safe_ik(base_matrix, q_guess)
        if q_target is None: q_target = q_guess

        self._latency_buffer.append(q_target)
        delay_steps = self._random_state.randint(1, self._latency_buffer.maxlen)
        idx = max(0, len(self._latency_buffer) - 1 - delay_steps)
        delayed_q_target = self._latency_buffer[idx]

        physics.set_control(delayed_q_target)

    def get_reward(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan": return NAN_PENALTY
        if not is_safe: return COLLISION_PENALTY

        real_tip_pos, real_wrist_mat = self._get_detector_tip_state(physics)

        target_pos = self._current_base_target[:3]
        target_euler = self._current_base_target[3:]
        target_rot = R.from_euler('xyz', target_euler, degrees=False)
        target_mat = target_rot.as_matrix()

        dist = np.linalg.norm(target_pos - real_tip_pos)

        r_diff = np.dot(real_wrist_mat, target_mat.T)
        trace = np.trace(r_diff)
        trace = np.clip(trace, -1.0, 3.0)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

        reward_pos = 10.0 * np.exp(-2000 * dist)
        reward_rot = 10.0 * np.exp(-1000 * angle_error)

        # 获取第5关节角度 (索引为4)
        q5 = physics.data.qpos[4]

        # 计算离奇异点的距离 (1.5708 rad = 90 deg)
        dist_to_singularity = abs(abs(q5) - 1.5708)

        # 如果太靠近奇异点 (比如小于 0.1弧度)，给巨大的惩罚
        if dist_to_singularity < 0.005:
            reward_singularity = -5.0 * (0.1 - dist_to_singularity)
            # print(q5)
        else:
            reward_singularity = 0.0

        return reward_pos * reward_rot + reward_singularity

    def get_termination(self, physics):
        is_safe, reason = self._check_collision_and_height(physics)

        if reason == "physics_nan" or (not is_safe and COLLISION_TERMINATE):
            return 1.0

        if self.path_index >= len(self.path_points) + 5:
            return 0.0

        return None

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        qpos = physics.data.qpos[:6].astype(np.float32)
        qvel = physics.data.qvel[:6].astype(np.float32)

        if np.isnan(qpos).any(): qpos[:] = 0.0
        if np.isnan(qvel).any(): qvel[:] = 0.0

        obs['qpos'] = qpos
        obs['qvel'] = qvel

        tip_pos, wrist_mat = self._get_detector_tip_state(physics)
        obs['ee_pos'] = tip_pos.astype(np.float32)

        if self._current_base_target is not None:
            target_pos = self._current_base_target[:3]
            pos_error = (target_pos - tip_pos).astype(np.float32)
            target_euler = self._current_base_target[3:]
            current_rot = R.from_matrix(wrist_mat)
            current_euler = current_rot.as_euler('xyz', degrees=False)
            rot_error = target_euler - current_euler
            rot_error = (rot_error + np.pi) % (2 * np.pi) - np.pi
            obs['tracking_error'] = np.concatenate([pos_error, rot_error]).astype(np.float32)
        else:
            obs['tracking_error'] = np.zeros(6, dtype=np.float32)

        return obs

    def _randomize_physics(self, physics):
        if hasattr(physics.model, 'dof_damping'):
            dof = physics.model.nv
            arm_dof = 6
            if dof >= arm_dof:
                physics.model.dof_damping[:arm_dof] *= self._random_state.uniform(0.9, 1.1, arm_dof)

    def _generate_scan_path(self, center):
        points = []
        scan_height = TARGET_SCAN_HEIGHT + self._random_state.uniform(-0.01, 0.01)
        scan_length = 0.2
        num_points = 150

        start_x = center[0] + self._random_state.uniform(-0.02, 0.02)
        start_y = center[1] + self._random_state.uniform(-0.02, 0.02)
        z = center[2] + scan_height

        for i in range(num_points):
            alpha = i / num_points
            x = start_x + 0.005 * np.sin(alpha * 10)
            y = start_y + (alpha - 0.5) * scan_length
            rot = [3.14, 0, 0]  # 垂直向下
            points.append(np.concatenate([[x, y, z], rot]))
        return points

    def _generate_mock_path_pattern(self):
        return self._generate_scan_path(SAMPLE_CENTER)


def load_env():
    scene_xml = "scene_with_sample.xml"
    robot_xml = "aubo_i5_withdetector.xml"

    if not os.path.exists(scene_xml): scene_xml = os.path.join("mjcf", scene_xml)
    if not os.path.exists(robot_xml): robot_xml = os.path.join("mjcf", robot_xml)

    if not os.path.exists(scene_xml): raise FileNotFoundError(f"Missing {scene_xml}")
    if not os.path.exists(robot_xml): raise FileNotFoundError(f"Missing {robot_xml}")

    task = AuboSceneScanTask(scene_xml_path=scene_xml, robot_xml_path=robot_xml)

    env = control.Environment(
        physics=mujoco.Physics.from_xml_path(scene_xml),
        task=task,
        time_limit=15.0,
        control_timestep=CONTROL_TIMESTEP
    )
    return env