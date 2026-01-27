import os
import sys

# ==============================================================================
# [底层修复 1] 环境变量清理
# ==============================================================================
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
    del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]
if "QT_PLUGIN_PATH" in os.environ:
    del os.environ["QT_PLUGIN_PATH"]

# ==============================================================================
# [底层修复 2] 优先导入 PyQt5
# ==============================================================================
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QRadioButton, QCheckBox, QFormLayout,
                             QFrame, QMessageBox, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap

# ==============================================================================
# [底层修复 3] 导入 MuJoCo 及科学计算库
# ==============================================================================
import time
import numpy as np
import trimesh
import pyvista as pv

pv.set_plot_theme('document')

import vtk
import mujoco
import mujoco.viewer
import glfw
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation as R

try:
    import curve_utils
    from casadi_ik import Kinematics
except ImportError:
    print("错误: 缺少依赖文件 curve_utils.py 或 casadi_ik.py")

# --- 常量定义 ---
SCENE_XML_PATH = os.path.join("mjcf", "scene_with_sample.xml")
SAMPLE_PATH = os.path.join("models", "sample.STL")
ROBOTICARM_MODEL_PATH = os.path.join("mjcf", "aubo_i5_withcam.xml")

END_JOINT = "wrist3_Link"
CAM_RES = (1280, 1280)
SAMPLE_OFFSET = [0.0, -0.49, 0.2]
TCP_OFFSET = np.array([0.0, 0.067, 0.0965])

# 【位置确认】ArUco 标定板位置
ARUCO_POS = [-0.236, -0.365, 0.212]


class ArUcoInjector:
    """
    辅助类：用于动态修改 MuJoCo XML，插入 ArUco 标定板
    """

    @staticmethod
    def inject_and_save(xml_path, position):
        if not os.path.exists(xml_path):
            if os.path.exists(os.path.basename(xml_path)):
                xml_path = os.path.basename(xml_path)
            else:
                raise FileNotFoundError(f"XML file not found: {xml_path}")

        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        asset_str = """
        <asset>
            <texture name="aruco_tex" type="2d" builtin="checker" rgb1="0.0 0.0 0.0" rgb2="1.0 1.0 1.0" width="512" height="512" mark="none"/>
            <material name="aruco_mat" texture="aruco_tex" texrepeat="4 4" reflectance="0"/>
        </asset>
        """

        # 【关键修复】添加 contype="0" conaffinity="0"
        # 这会关闭该物体的物理碰撞检测，使其成为“幽灵”物体
        # 即使它插入到桌子内部，也不会产生排斥力炸飞桌子
        body_str = f"""
        <body name="aruco_board" pos="{position[0]} {position[1]} {position[2]}" euler="0 0 0">
            <geom type="box" size="0.06 0.06 0.001" material="aruco_mat" contype="0" conaffinity="0"/>
            <geom type="box" size="0.065 0.065 0.0005" rgba="0.1 0.1 0.1 1" pos="0 0 -0.001" contype="0" conaffinity="0"/>
        </body>
        """

        if "</worldbody>" in xml_content:
            xml_content = xml_content.replace("</worldbody>", body_str + "\n</worldbody>")

        if "</mujoco>" in xml_content:
            if "<worldbody>" in xml_content:
                xml_content = xml_content.replace("<worldbody>", asset_str + "\n<worldbody>")
            else:
                xml_content = xml_content.replace("</mujoco>", asset_str + "\n</mujoco>")

        dirname = os.path.dirname(xml_path)
        basename = os.path.basename(xml_path)
        temp_filename = f"temp_aruco_{basename}"
        temp_path = os.path.join(dirname, temp_filename)

        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        print(f"[ArUcoInjector] Temporary scene file created: {temp_path}")
        return temp_path


class EnhancedSimulator:
    def __init__(self, xml_path, end_joint, robot_model_path, cam_res=(1280, 1280)):
        self.xml_path = xml_path
        self.end_joint = end_joint
        self.robot_model_path = robot_model_path
        self.cam_res = cam_res

        self.paused = True
        self.use_rl = False
        self.rl_model = None
        self.calibration_mode = False
        self.temp_xml_path = None

        self.physics_dt = 0.002
        self.control_dt = 0.02
        self.n_substeps = int(self.control_dt / self.physics_dt)
        self.scan_interval = 0.5

        self.path_points = None
        self.path_normals = None
        self.current_idx = 0
        self.last_scan_time = 0
        self.scan_height = 0.1
        self.target_qpos = None

        self.T_target_cache = None
        self.T_twist_cache = None

        self.load_rl_agent()
        self.init_mujoco()

        try:
            self.ik_solver = Kinematics(self.end_joint)
            self.ik_solver.buildFromMJCF(self.robot_model_path)
        except Exception as e:
            print(f"IK 初始化失败: {e}")

        self.init_offscreen()
        print("启动 MuJoCo Passive Viewer...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def load_rl_agent(self):
        try:
            from stable_baselines3 import PPO
            self.HAS_SB3 = True
        except ImportError:
            self.HAS_SB3 = False
            print("警告: 未安装 stable_baselines3，RL 功能将不可用。")
            return

        candidates = ["aubo_scene_scan_final.zip", "logs/scene_rl/best_model.zip"]
        model_path = None
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

        if model_path:
            try:
                self.rl_model = PPO.load(model_path, device='cpu')
                print(f"RL 模型已加载: {model_path}")
            except Exception as e:
                print(f"RL 模型加载错误: {e}")
        else:
            print("未找到 RL 模型文件，仅使用 IK 模式。")

    def init_mujoco(self):
        try:
            self.temp_xml_path = ArUcoInjector.inject_and_save(self.xml_path, ARUCO_POS)
            self.model = mujoco.MjModel.from_xml_path(self.temp_xml_path)
            self.data = mujoco.MjData(self.model)

            self.model.opt.timestep = self.physics_dt
            self.target_qpos = self.data.qpos[:6].copy()
            self.data.ctrl[:6] = self.target_qpos
            mujoco.mj_step(self.model, self.data)

        except Exception as e:
            print(f"MuJoCo 加载失败: {e}")
            if self.temp_xml_path and os.path.exists(self.temp_xml_path):
                os.remove(self.temp_xml_path)
            raise e

    def init_offscreen(self):
        if not glfw.init(): raise Exception("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, 0)
        self.offscreen_window = glfw.create_window(self.cam_res[0], self.cam_res[1], "Offscreen", None, None)
        glfw.make_context_current(self.offscreen_window)

        self.gl_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.viewport = mujoco.MjrRect(0, 0, self.cam_res[0], self.cam_res[1])
        self.rgb_buffer = np.zeros((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8)

        self.cam_robot = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if cam_id != -1:
            self.cam_robot.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam_robot.fixedcamid = cam_id

        self.cam_global = mujoco.MjvCamera()
        self.cam_global.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_global.lookat = [0.1, -0.1, 0.4]
        self.cam_global.distance = 1.8
        self.cam_global.elevation = -30
        self.cam_global.azimuth = 135

    def compute_target_matrix(self, position, normal, standoff=0.1):
        target_pos = position + normal * standoff
        z_axis = -normal / np.linalg.norm(normal)
        ref_axis = np.array([1, 0, 0])
        if np.abs(np.dot(z_axis, ref_axis)) > 0.9: ref_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, ref_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        T = np.eye(4)
        T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        T[:3, 3] = target_pos
        return T

    def step(self):
        # 1. 物理/逻辑更新层
        if not self.paused:
            if self.path_points is not None:
                now = time.time()
                if not self.calibration_mode:
                    if now - self.last_scan_time > self.scan_interval:
                        self.current_idx = (self.current_idx + 1) % len(self.path_points)
                        self.last_scan_time = now
                else:
                    if now - self.last_scan_time > 0.02:
                        if self.current_idx < len(self.path_points) - 1:
                            self.current_idx += 1
                            self.last_scan_time = now

            self.perform_ik_step()

            for _ in range(self.n_substeps):
                if self.target_qpos is not None:
                    self.data.ctrl[:6] = self.target_qpos
                mujoco.mj_step(self.model, self.data)

        # 2. 渲染同步层 (强制刷新，防止暂停时标记消失)
        if self.viewer.is_running():
            self.viewer.user_scn.ngeom = 0
            self._add_markers(self.viewer.user_scn)
            self.viewer.sync()

    def perform_ik_step(self):
        if self.path_points is None: return

        target_pt = self.path_points[self.current_idx]

        if len(target_pt) == 6:
            pos = target_pt[:3]
            if self.calibration_mode:
                rot = R.from_euler('xyz', target_pt[3:], degrees=False)
                T_target = np.eye(4)
                T_target[:3, :3] = rot.as_matrix()
                T_target[:3, 3] = pos
            else:
                nm = np.array([0, 0, 1])
                T_target = self.compute_target_matrix(pos, nm, self.scan_height)
        else:
            pos = target_pt
            if self.path_normals is not None:
                nm = self.path_normals[self.current_idx]
            else:
                nm = np.array([0, 0, 1])
            T_target = self.compute_target_matrix(pos, nm, self.scan_height)

        self.T_target_cache = T_target
        T_wrist = T_target.copy()
        T_wrist[:3, 3] -= T_wrist[:3, :3] @ TCP_OFFSET
        self.T_twist_cache = T_wrist

        init_q = self.data.qpos[:6].copy()
        try:
            q_sol, info = self.ik_solver.ik(T_wrist, current_arm_motor_q=init_q)
            if info["success"]:
                self.target_qpos = np.clip(q_sol, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
            else:
                if self.calibration_mode:
                    print(f"[IK Warning] Frame {self.current_idx} IK failed.")
        except Exception as e:
            pass

    def start_calibration(self):
        self.calibration_mode = True
        self.paused = True

        print(f"[Calibration] Target ArUco Pos: {ARUCO_POS}")

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if site_id != -1:
            current_tip = self.data.site_xpos[site_id].copy()
        else:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.end_joint)
            if body_id != -1:
                current_tip = self.data.xpos[body_id].copy()
            else:
                current_tip = self.data.xpos[6].copy()

        hover_pos = np.array(ARUCO_POS) + np.array([0, 0, 0.25])

        steps = 50
        path = []
        for i in range(steps):
            alpha = i / (steps - 1)
            pos = current_tip * (1 - alpha) + hover_pos * alpha
            pt_6d = np.concatenate([pos, [3.14, 0, 0]])
            path.append(pt_6d)

        self.path_points = path
        self.path_normals = None
        self.current_idx = 0
        self.last_scan_time = time.time()

        self.paused = False
        print(">>> 进入标定模式，开始移动...")

    def set_path(self, points, normals, height=0.1):
        self.calibration_mode = False
        self.path_points = points
        self.path_normals = normals
        self.scan_height = height
        self.current_idx = 0
        self.last_scan_time = time.time()

        self.paused = True

        self.perform_ik_step()
        if self.target_qpos is not None:
            self.data.qpos[:6] = self.target_qpos
            self.data.qvel[:6] = 0
            mujoco.mj_forward(self.model, self.data)
            if self.viewer: self.viewer.sync()

    def manual_adjust(self, direction):
        if self.path_points is None: return
        self.current_idx = (self.current_idx + direction) % len(self.path_points)
        self.perform_ik_step()
        if self.target_qpos is not None:
            self.data.qpos[:6] = self.target_qpos
            mujoco.mj_forward(self.model, self.data)
            if self.viewer: self.viewer.sync()

    def render_offscreen(self):
        glfw.make_context_current(self.offscreen_window)

        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_robot,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers(self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.gl_context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.gl_context)
        img_robot = np.flipud(self.rgb_buffer).copy()

        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.cam_global,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self._add_markers(self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.gl_context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.gl_context)
        img_global = np.flipud(self.rgb_buffer).copy()

        return img_robot, img_global

    def _add_markers(self, scene):
        if scene.ngeom + 3 >= scene.maxgeom: return
        # 绘制 ArUco 位置
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.002],
            pos=ARUCO_POS,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 0.5]
        )
        scene.ngeom += 1

        if self.path_points is not None and len(self.path_points) > 0:
            # 当前点 (红色小球)
            pt = self.path_points[self.current_idx]
            pos = pt[:3] if len(pt) >= 3 else pt

            # 增加 3mm 的 Z 轴偏移，防止Z-fighting
            display_pos = pos.copy()
            display_pos[2] += 0.003

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.006, 0, 0],
                pos=display_pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]
            )
            scene.ngeom += 1

            # 目标点 (绿色小球)
            if self.T_target_cache is not None:
                pt_tcp = self.T_target_cache[:3, 3]
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.003, 0, 0],
                    pos=pt_tcp,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 0.7]
                )
                scene.ngeom += 1

    def close(self):
        if self.temp_xml_path and os.path.exists(self.temp_xml_path):
            try:
                os.remove(self.temp_xml_path)
            except:
                pass

        if self.viewer: self.viewer.close()
        if self.offscreen_window: glfw.destroy_window(self.offscreen_window)
        glfw.terminate()


# --- GUI 主类 ---

class RobotPathInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)

    def right_button_press(self, obj, event): self.StartPan()

    def right_button_release(self, obj, event): self.EndPan()


class SquareLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

    def heightForWidth(self, width): return width

    def sizeHint(self): return self.size()

    def resizeEvent(self, event):
        target_width = self.width()
        if self.height() != target_width: self.setFixedHeight(target_width)
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Scan System (RL & Calibration Enhanced)")
        self.resize(1650, 1050)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 10pt; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #333; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; left: 10px; color: #4facfe; }
            QLineEdit { background-color: #404040; border: 1px solid #555; border-radius: 3px; padding: 4px; color: #fff; }
            QPushButton { background-color: #0078d4; border: none; border-radius: 4px; padding: 8px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #1084e0; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QLabel { color: #ccc; }
        """)

        self.stl_path = SAMPLE_PATH
        self.trimesh_obj = None
        self.current_points = []
        self.step_size = 10.0
        self.sim = None

        self.render_skip_counter = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        try:
            self.sim = EnhancedSimulator(SCENE_XML_PATH, END_JOINT, ROBOTICARM_MODEL_PATH, CAM_RES)
        except Exception as e:
            print(f"Simulation Init Error: {e}")

        self.init_ui()
        self.load_model()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(25)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_settings_panel())
        left_layout.addWidget(self.create_advanced_panel())
        left_layout.addWidget(self.create_control_panel())
        left_layout.addStretch()

        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(400)
        layout.addWidget(left_container)

        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)

        cam_box = QWidget()
        c_layout = QHBoxLayout(cam_box)
        self.lbl_cam_robot = SquareLabel("Robot Camera")
        self.lbl_cam_global = SquareLabel("Simulation View")
        c_layout.addWidget(self.lbl_cam_robot)
        c_layout.addWidget(self.lbl_cam_global)
        r_layout.addWidget(cam_box, 0)

        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#e1e1e1")
        r_layout.addWidget(self.plotter, 1)

        layout.addWidget(right_panel)

    def create_settings_panel(self):
        frame = QFrame()
        vbox = QVBoxLayout(frame)

        grp = QGroupBox("模式选择")
        v = QVBoxLayout()
        self.rb_spiral = QRadioButton("Spiral (螺旋扫描)")
        self.rb_spiral.setChecked(True)
        self.rb_zigzag = QRadioButton("Zigzag (弓字扫描)")
        self.rb_spiral.toggled.connect(self.update_inputs)
        v.addWidget(self.rb_spiral)
        v.addWidget(self.rb_zigzag)
        grp.setLayout(v)
        vbox.addWidget(grp)

        grp_p = QGroupBox("基本参数")
        f = QFormLayout()
        self.inp_step = QLineEdit("10.0")
        self.inp_z_thresh = QLineEdit("0.2")
        self.inp_radius = QLineEdit("150.0")
        self.chk_center = QCheckBox("自动中心")
        self.chk_center.setChecked(True)
        self.inp_cx = QLineEdit("0.0")
        self.inp_cx.setEnabled(False)
        self.inp_cy = QLineEdit("0.0")
        self.inp_cy.setEnabled(False)
        self.chk_center.toggled.connect(self.update_inputs)
        self.inp_interval = QLineEdit("500")
        self.inp_height = QLineEdit("0.1")

        f.addRow("扫描步长(mm):", self.inp_step)
        f.addRow("法向Z阈值:", self.inp_z_thresh)
        f.addRow("最大半径(mm):", self.inp_radius)
        f.addRow("扫描间隔(ms):", self.inp_interval)
        f.addRow("扫描高度(m):", self.inp_height)
        f.addRow(self.chk_center)
        f.addRow("Center X:", self.inp_cx)
        f.addRow("Center Y:", self.inp_cy)
        grp_p.setLayout(f)
        vbox.addWidget(grp_p)

        grp_roi = QGroupBox("ROI 区域限制 (不限留空)")
        g_roi = QFormLayout()
        self.inp_xmin = QLineEdit()
        self.inp_xmax = QLineEdit()
        self.inp_ymin = QLineEdit()
        self.inp_ymax = QLineEdit()
        self.inp_zmin = QLineEdit()
        self.inp_zmax = QLineEdit()

        for w in [self.inp_xmin, self.inp_xmax, self.inp_ymin, self.inp_ymax, self.inp_zmin, self.inp_zmax]:
            w.setPlaceholderText("不限")
            w.setFixedWidth(60)

        def make_row(lbl, w1, w2):
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            h.addWidget(w1)
            h.addWidget(QLabel("~"))
            h.addWidget(w2)
            return h

        g_roi.addRow(make_row("X:", self.inp_xmin, self.inp_xmax))
        g_roi.addRow(make_row("Y:", self.inp_ymin, self.inp_ymax))
        g_roi.addRow(make_row("Z:", self.inp_zmin, self.inp_zmax))
        grp_roi.setLayout(g_roi)
        vbox.addWidget(grp_roi)

        btn_gen = QPushButton("生成路径并发送")
        btn_gen.clicked.connect(self.generate_and_send)
        btn_gen.setStyleSheet("background-color: #2ecc71; height: 40px;")
        vbox.addWidget(btn_gen)

        return frame

    def create_advanced_panel(self):
        frame = QFrame()
        vbox = QVBoxLayout(frame)

        grp_rl = QGroupBox("智能控制 (RL)")
        v_rl = QVBoxLayout()
        self.chk_rl = QCheckBox("启用 RL 避障/控制")
        self.chk_rl.setToolTip("启用后，将使用训练好的神经网络代替纯 IK")
        self.chk_rl.setEnabled(False)
        self.chk_rl.setText("RL 正在加载...")

        self.rl_check_timer = QTimer(self)
        self.rl_check_timer.timeout.connect(self.check_rl_loaded)
        self.rl_check_timer.start(1000)

        self.chk_rl.toggled.connect(self.toggle_rl)
        v_rl.addWidget(self.chk_rl)
        grp_rl.setLayout(v_rl)
        vbox.addWidget(grp_rl)

        grp_calib = QGroupBox("系统标定")
        v_cal = QVBoxLayout()
        # lbl_info = QLabel(f"ArUco 位置: {ARUCO_POS}")
        # lbl_info.setStyleSheet("font-size: 8pt; color: #888;")
        # v_cal.addWidget(lbl_info)

        self.btn_calib = QPushButton("执行手眼标定动作")
        self.btn_calib.clicked.connect(self.start_calibration)
        self.btn_calib.setStyleSheet("background-color: #9b59b6;")
        v_cal.addWidget(self.btn_calib)
        grp_calib.setLayout(v_cal)
        vbox.addWidget(grp_calib)

        return frame

    def check_rl_loaded(self):
        if self.sim:
            if hasattr(self.sim, 'rl_model') and self.sim.rl_model is not None:
                self.chk_rl.setEnabled(True)
                self.chk_rl.setText("启用 RL 避障/控制")
                self.rl_check_timer.stop()
            elif hasattr(self.sim, 'HAS_SB3') and self.sim.HAS_SB3 is False:
                self.chk_rl.setText("RL 库未安装")
                self.rl_check_timer.stop()

    def create_control_panel(self):
        frame = QFrame()
        frame.setStyleSheet("background-color: #222; border-radius: 8px;")
        vbox = QVBoxLayout(frame)

        vbox.addWidget(QLabel("Simulation Status"))
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #0f0; font-family: Consolas;")
        vbox.addWidget(self.lbl_status)

        self.lbl_progress = QLabel("Point: 0 / 0")
        self.lbl_progress.setStyleSheet("font-size: 11pt; color: yellow;")
        vbox.addWidget(self.lbl_progress)

        self.btn_pause = QPushButton("开始/暂停 (Space)")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setStyleSheet("background-color: #e67e22;")
        vbox.addWidget(self.btn_pause)

        return frame

    def update_inputs(self):
        self.inp_cx.setEnabled(not self.chk_center.isChecked())
        self.inp_cy.setEnabled(not self.chk_center.isChecked())

    def get_float(self, line_edit, default=None):
        try:
            txt = line_edit.text()
            if not txt and default is not None: return default
            return float(txt)
        except:
            return default

    def get_roi_dict(self):
        return {
            'x': {'min': self.get_float(self.inp_xmin), 'max': self.get_float(self.inp_xmax)},
            'y': {'min': self.get_float(self.inp_ymin), 'max': self.get_float(self.inp_ymax)},
            'z': {'min': self.get_float(self.inp_zmin), 'max': self.get_float(self.inp_zmax)}
        }

    def load_model(self):
        if not os.path.exists(self.stl_path): return
        self.trimesh_obj = curve_utils.CurvePathPlanner.ensure_single_mesh(trimesh.load(self.stl_path))
        self.trimesh_obj.apply_scale(1000.0)
        self.pv_mesh = pv.read(self.stl_path)
        self.pv_mesh.points *= 1000.0

        c = self.trimesh_obj.bounding_box.centroid
        self.inp_cx.setText(f"{c[0]:.2f}")
        self.inp_cy.setText(f"{c[1]:.2f}")

        self.plotter.clear()
        self.plotter.add_mesh(self.pv_mesh, color="white", opacity=0.6, show_edges=False)
        try:
            self.plotter.add_camera_orientation_widget()
        except:
            self.plotter.add_axes()
        self.plotter.add_mesh(pv.Box(self.pv_mesh.bounds), color='grey', style='wireframe', opacity=0.3)
        self.plotter.view_isometric()
        self.plotter.iren.interactor.SetInteractorStyle(RobotPathInteractorStyle())

    def get_surface_point(self, x, y):
        ray_origin = np.array([[x, y, self.trimesh_obj.bounds[1, 2] + 50]])
        ray_dir = np.array([[0, 0, -1]])
        locs, _, _ = self.trimesh_obj.ray.intersects_location(ray_origin, ray_dir, multiple_hits=False)
        if len(locs) > 0: return locs[0]
        return np.array([x, y, self.trimesh_obj.bounds[1, 2]])

    def generate_and_send(self):
        if not self.trimesh_obj: return
        self.step_size = self.get_float(self.inp_step, 10.0)
        z_thresh = self.get_float(self.inp_z_thresh, 0.2)

        if self.sim:
            try:
                ms = float(self.inp_interval.text())
                self.sim.scan_interval = ms / 1000.0
            except:
                pass

        if self.rb_spiral.isChecked():
            radius = self.get_float(self.inp_radius, 150.0)
            if self.chk_center.isChecked():
                c = self.trimesh_obj.bounding_box.centroid
                cx, cy = c[0], c[1]
            else:
                cx, cy = self.get_float(self.inp_cx, 0.0), self.get_float(self.inp_cy, 0.0)

            center_pt = self.get_surface_point(cx, cy)
            display_pt = center_pt + np.array([0, 0, 0.5])
            self.plotter.clear()
            self.plotter.add_mesh(self.pv_mesh, color="white", opacity=0.6, show_edges=False)
            self.plotter.add_points(display_pt, color="red", point_size=15, render_points_as_spheres=True,
                                    label="Center")

            points, normals = curve_utils.CurvePathPlanner.compute_spiral_3d(
                self.trimesh_obj, cx, cy, radius, self.step_size, z_thresh
            )
        else:
            self.plotter.clear()
            self.plotter.add_mesh(self.pv_mesh, color="white", opacity=0.6, show_edges=False)
            points, normals = curve_utils.CurvePathPlanner.generate_zigzag_path(
                self.trimesh_obj, self.step_size, z_thresh
            )

        if len(points) > 0:
            roi_dict = self.get_roi_dict()
            points, normals = curve_utils.CurvePathPlanner.filter_by_roi(points, normals, roi_dict)

            b = self.trimesh_obj.bounds
            bounds = [
                roi_dict['x']['min'] if roi_dict['x']['min'] is not None else b[0, 0],
                roi_dict['x']['max'] if roi_dict['x']['max'] is not None else b[1, 0],
                roi_dict['y']['min'] if roi_dict['y']['min'] is not None else b[0, 1],
                roi_dict['y']['max'] if roi_dict['y']['max'] is not None else b[1, 1],
                roi_dict['z']['min'] if roi_dict['z']['min'] is not None else b[0, 2],
                roi_dict['z']['max'] if roi_dict['z']['max'] is not None else b[1, 2]
            ]
            self.plotter.add_mesh(pv.Box(bounds), color="green", style='wireframe', opacity=0.3, line_width=2)

        self.current_points = points

        try:
            self.plotter.add_camera_orientation_widget()
        except:
            self.plotter.add_axes()
        self.plotter.add_mesh(pv.Box(self.pv_mesh.bounds), color='grey', style='wireframe', opacity=0.3)

        if len(points) > 0:
            line = pv.lines_from_points(points)
            line["scalars"] = np.arange(len(points))
            self.plotter.add_mesh(line, cmap="turbo", line_width=3, show_scalar_bar=False)

            self.plotter.add_mesh(points, scalars=points[:, 2], cmap="viridis",
                                  point_size=6, render_points_as_spheres=True, show_scalar_bar=False)

            cone = pv.Cone(radius=0.04, height=0.15, direction=(1, 0, 0))
            pd = pv.PolyData(points)
            pd["normals"] = normals
            glyphs = pd.glyph(scale=False, orient="normals", geom=cone, factor=self.step_size * 0.6)
            self.plotter.add_mesh(glyphs, color="#dddddd", opacity=0.6)

            self.plotter.add_mesh(
                pv.Sphere(radius=self.step_size * 0.4, center=points[0]),
                color="red", name="Highlight", render_points_as_spheres=True
            )

        if self.sim:
            scan_h = self.get_float(self.inp_height, 0.1)
            offset = np.array(SAMPLE_OFFSET)
            pts_m = points / 1000.0 + offset

            self.sim.set_path(pts_m, normals, height=scan_h)

            self.lbl_progress.setText(f"Path Generated: {len(points)} points")
            self.activateWindow()
            self.setFocus()

    def toggle_rl(self):
        if self.sim:
            self.sim.use_rl = self.chk_rl.isChecked()
            mode = "RL Agent" if self.sim.use_rl else "Kinematic IK"
            self.lbl_status.setText(f"Control Mode: {mode}")

    def start_calibration(self):
        if self.sim:
            self.sim.start_calibration()
            self.lbl_status.setText("Mode: Calibration")

    def toggle_pause(self):
        if self.sim:
            self.sim.paused = not self.sim.paused

    def game_loop(self):
        if not self.sim: return

        self.sim.step()

        self.render_skip_counter += 1
        if self.render_skip_counter % 2 != 0: return

        try:
            img_r, img_g = self.sim.render_offscreen()

            def np2pixmap(img):
                h, w, c = img.shape
                qimg = QImage(img.data.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
                return QPixmap.fromImage(qimg)

            self.lbl_cam_robot.setPixmap(np2pixmap(img_r))
            self.lbl_cam_global.setPixmap(np2pixmap(img_g))

            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter * 2
                self.fps_counter = 0
                self.last_fps_time = time.time()

            state = "PAUSED" if self.sim.paused else "RUNNING"
            mode = "CALIB" if self.sim.calibration_mode else ("RL" if self.sim.use_rl else "IK")
            idx = self.sim.current_idx
            total = len(self.sim.path_points) if self.sim.path_points is not None else 0

            self.lbl_status.setText(f"FPS:{self.current_fps} | [{state}] {mode}")
            self.lbl_progress.setText(f"Point: {idx + 1} / {total}")

            if len(self.current_points) > idx and not self.sim.calibration_mode:
                current_pos = self.current_points[idx]
                self.plotter.add_mesh(
                    pv.Sphere(radius=self.step_size * 0.4, center=current_pos),
                    color="red", name="Highlight", render_points_as_spheres=True,
                    reset_camera=False
                )

        except Exception:
            pass

    def keyPressEvent(self, event):
        if not self.sim: return
        if event.key() == Qt.Key_Left:
            self.sim.manual_adjust(-1)
        elif event.key() == Qt.Key_Right:
            self.sim.manual_adjust(1)
        elif event.key() == Qt.Key_Space:
            self.toggle_pause()

    def closeEvent(self, event):
        self.timer.stop()
        if self.sim: self.sim.close()
        QApplication.quit()
        event.accept()


if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())