import os
import sys

# 解决某些 Linux/Conda 环境下 Qt 插件路径冲突问题（可尝试注释使用）
# if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
#     del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

import time
import numpy as np
import trimesh
import pyvista as pv
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QRadioButton, QCheckBox, QFormLayout,
                             QFrame, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from pyvistaqt import QtInteractor

import curve_utils
from mjplayground_sim import MuJoCoSimulator

SCENE_XML_PATH = "mjcf/scene_with_sample.xml"  # 场景 XML 包含所有 MJCF 所需文件
SAMPLE_PATH = "models/sample.STL"   # 待扫描样品模型
END_JOINT = "wrist3_Link"    # 末端执行器名称
ROBOTICARM_MODEL_PATH = "mjcf/aubo_i5_withcam.xml"    # 机械臂模型 MJCF/urdf 文件
CAM_RES =(1280, 1280)    # 同步摄像头分辨率。 是清晰度与FPS的平衡点
SAMPLE_OFFSET = [0.0, -0.49, 0.2]   # 仿真场景的坐标偏移，需根据 xml 调整
# [-0.25, -0.6, 0.2] curve_model
# [0.0, -0.49, 0.2] sample


class RobotPathInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    自定义界面
    功能：选取spiral/zigzag扫描模式；
         设置扫描步长和等高筛选阈值；
         设置扫描参数及框选区域；
         开关仿真自动扫描；
         仿真实时可视化显示及单步切换
    """

    def __init__(self, parent=None):
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)

    def right_button_press(self, obj, event):
        self.StartPan()

    def right_button_release(self, obj, event):
        self.EndPan()


class SquareLabel(QLabel):

    # 自定义 Label 控件，保持宽高比，并使内部图片自动缩放填充
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)  # 图片自动缩放填满控件
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许随布局拉伸
        self.setMinimumSize(100, 100)  # 设置最小尺寸

    def heightForWidth(self, width):
        return width  # 锁定正方形

    def sizeHint(self):
        return self.size()

    def resizeEvent(self, event):
        # 当宽度变化时，强制更新高度
        target_width = self.width()
        if self.height() != target_width:
            self.setFixedHeight(target_width)
        super().resizeEvent(event)


class DarkStyle:
    # 应用程序深色主题样式表，定义了窗口、按钮、输入框的配色方案
    SHEET = """
    QMainWindow { background-color: #2b2b2b; }
    QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 10pt; }
    QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #333; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; left: 10px; color: #4facfe; }
    QLineEdit { background-color: #404040; border: 1px solid #555; border-radius: 3px; padding: 4px; color: #fff; }
    QPushButton { background-color: #0078d4; border: none; border-radius: 4px; padding: 8px; color: white; font-weight: bold; }
    QPushButton:hover { background-color: #1084e0; }
    QLabel { color: #ccc; }
    """


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Scan System Integration")
        self.resize(1600, 1000)
        self.setStyleSheet(DarkStyle.SHEET)

        # 数据模型
        self.stl_path = SAMPLE_PATH
        self.trimesh_obj = None
        self.current_points = []
        self.current_normals = []
        self.step_size = 10.0

        # 性能优化计数器
        self.render_skip_counter = 0  # 用于 GUI 渲染跳帧

        # 初始化仿真后端
        self.sim = None
        try:
            self.sim = MuJoCoSimulator(SCENE_XML_PATH, END_JOINT, ROBOTICARM_MODEL_PATH, CAM_RES)
        except Exception as e:
            print(f"仿真初始化错误: {e}")

        # 定时器循环
        # 33ms 约等于 30FPS 的触发频率
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(25)

        # FPS 统计
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # 初始化界面与加载模型
        self.init_ui()
        self.load_model()

        # 允许窗口响应键盘事件 (如空格暂停)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def init_ui(self):
        # 构建主窗口布局
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 1. 左侧控制栏 (设置参数、控制仿真)
        left_layout = QVBoxLayout()
        settings_panel = self.create_settings_panel()
        control_panel = self.create_control_display()
        left_layout.addWidget(settings_panel)
        left_layout.addWidget(control_panel)
        left_layout.addStretch()

        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(380)
        layout.addWidget(left_container)

        # 2. 右侧显示面板 (摄像头 + 路径预览)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 2.1 上方：两个正方形摄像头监控
        cam_container = QWidget()
        cam_layout = QHBoxLayout(cam_container)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(10)

        # 使用自定义的 SquareLabel 确保显示
        self.lbl_cam_robot = SquareLabel("Robot Camera")
        self.lbl_cam_global = SquareLabel("Simulation View")

        cam_layout.addWidget(self.lbl_cam_robot)
        cam_layout.addWidget(self.lbl_cam_global)

        right_layout.addWidget(cam_container, stretch=0)  # stretch=0 表示不自动拉伸高度

        # 2.2 下方：路径规划 3D 预览
        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#e1e1e1")
        right_layout.addWidget(self.plotter, stretch=1)  # stretch=1 表示填充剩余空间

        layout.addWidget(right_panel)

    def create_settings_panel(self):
        # 创建左侧参数设置面板
        frame = QFrame()
        vbox = QVBoxLayout(frame)

        # 模式选择
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

        # 扫描参数
        grp_p = QGroupBox("基本参数")
        f = QFormLayout()
        self.inp_step = QLineEdit("10.0")  # 采样点间距
        self.inp_z_thresh = QLineEdit("0.2")  # 法向Z阈值 (0-1)
        self.inp_radius = QLineEdit("150.0")  # 螺旋最大半径
        self.chk_center = QCheckBox("自动计算中心")
        self.chk_center.setChecked(True)
        self.inp_cx = QLineEdit("0.0")  # 手动指定中心 X
        self.inp_cy = QLineEdit("0.0")  # 手动指定中心 Y
        self.chk_center.toggled.connect(self.update_inputs)
        self.inp_interval = QLineEdit("500")  # 仿真每个点的停留时间 (ms)
        self.inp_height = QLineEdit("0.1")  # 扫描悬停高度 (m)

        f.addRow("扫描步长(mm):", self.inp_step)
        f.addRow("法向Z阈值:", self.inp_z_thresh)
        f.addRow("最大半径(mm):", self.inp_radius)
        f.addRow("扫描间隔(ms):", self.inp_interval)
        f.addRow("扫描高度(m):", self.inp_height)
        f.addRow(self.chk_center)
        f.addRow("X:", self.inp_cx)
        f.addRow("Y:", self.inp_cy)
        grp_p.setLayout(f)
        vbox.addWidget(grp_p)

        # ROI 区域限制
        grp_roi = QGroupBox("ROI 区域限制 (不限留空)")
        g_roi = QFormLayout()
        self.inp_xmin = QLineEdit()
        self.inp_xmax = QLineEdit()
        self.inp_ymin = QLineEdit()
        self.inp_ymax = QLineEdit()
        self.inp_zmin = QLineEdit()  # 可添加默认 Z 轴最小值过滤底板
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

        # 生成扫描路径按钮
        btn_gen = QPushButton("生成路径并发送至仿真")
        btn_gen.clicked.connect(self.generate_and_send)
        btn_gen.setFixedHeight(50)
        vbox.addStretch()
        vbox.addWidget(btn_gen)

        return frame

    def create_control_display(self):
        # 创建仿真控制状态面板
        frame = QFrame()
        frame.setStyleSheet("background-color: #222; border-radius: 8px;")
        vbox = QVBoxLayout(frame)

        lbl = QLabel("Simulation Control")
        lbl.setStyleSheet("font-size: 12pt; font-weight: bold; color: #4facfe;")
        lbl.setAlignment(Qt.AlignCenter)
        vbox.addWidget(lbl)

        self.lbl_stats = QLabel("Running...")
        self.lbl_stats.setStyleSheet("font-family: Consolas; font-size: 10pt; color: #0f0;")
        vbox.addWidget(self.lbl_stats)

        self.lbl_progress = QLabel("Point: 0 / 0")
        self.lbl_progress.setStyleSheet("font-size: 11pt; color: yellow;")
        vbox.addWidget(self.lbl_progress)

        h = QHBoxLayout()
        self.btn_pause = QPushButton("开始/暂停 (Space)")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setStyleSheet("background-color: #e67e22; height: 35px;")
        h.addWidget(self.btn_pause)
        vbox.addLayout(h)

        lbl_hint = QLabel("交互说明:\n[←/→] 切换点位\n[Space] 暂停/继续\n[鼠标左键] 旋转\n[鼠标右键] 平移\n[滚轮] 缩放")
        lbl_hint.setStyleSheet("color: #888; font-size: 9pt; margin-top: 10px;")
        vbox.addWidget(lbl_hint)

        return frame

    def update_inputs(self):
        # 根据复选框状态启用/禁用输入框
        self.inp_cx.setEnabled(not self.chk_center.isChecked())
        self.inp_cy.setEnabled(not self.chk_center.isChecked())

    def get_float(self, line_edit, default=None):
        # 安全解析浮点数输入
        try:
            txt = line_edit.text()
            if not txt and default is not None: return default
            return float(txt)
        except:
            return default

    def get_roi_dict(self):
        # 收集 ROI 输入框参数
        return {
            'x': {'min': self.get_float(self.inp_xmin), 'max': self.get_float(self.inp_xmax)},
            'y': {'min': self.get_float(self.inp_ymin), 'max': self.get_float(self.inp_ymax)},
            'z': {'min': self.get_float(self.inp_zmin), 'max': self.get_float(self.inp_zmax)}
        }

    def load_model(self):
        # 加载 STL 模型在窗口显示
        if not os.path.exists(self.stl_path): return
        self.trimesh_obj = curve_utils.CurvePathPlanner.ensure_single_mesh(trimesh.load(self.stl_path))
        self.trimesh_obj.apply_scale(1000.0)  # 模型单位转换至mm
        self.pv_mesh = pv.read(self.stl_path)
        self.pv_mesh.points *= 1000.0  # mesh单位转换至mm

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
        style = RobotPathInteractorStyle()
        self.plotter.iren.interactor.SetInteractorStyle(style)

    def get_surface_point(self, x, y):
        # 计算 (x,y) 对应的模型表面 Z 坐标
        ray_origin = np.array([[x, y, self.trimesh_obj.bounds[1, 2] + 50]])
        ray_dir = np.array([[0, 0, -1]])
        locs, _, _ = self.trimesh_obj.ray.intersects_location(ray_origin, ray_dir, multiple_hits=False)
        if len(locs) > 0: return locs[0]
        return np.array([x, y, self.trimesh_obj.bounds[1, 2]])

    def generate_and_send(self):
        # 生成扫描路径并发送给仿真后端
        if not self.trimesh_obj: return
        self.step_size = self.get_float(self.inp_step, 10.0)
        z_thresh = self.get_float(self.inp_z_thresh, 0.2)

        # 更新仿真间隔
        if self.sim:
            try:
                ms = float(self.inp_interval.text())
                self.sim.scan_interval = ms / 1000.0
            except:
                pass

        # 1. 路径生成逻辑 (Spiral 或 Zigzag)
        if self.rb_spiral.isChecked():
            radius = self.get_float(self.inp_radius, 150.0)
            if self.chk_center.isChecked():
                c = self.trimesh_obj.bounding_box.centroid
                cx, cy = c[0], c[1]
            else:
                cx, cy = self.get_float(self.inp_cx, 0.0), self.get_float(self.inp_cy, 0.0)

            # 标记中心点
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

        # 2. 应用 ROI 过滤
        if len(points) > 0:
            roi_dict = self.get_roi_dict()
            points, normals = curve_utils.CurvePathPlanner.filter_by_roi(points, normals, roi_dict)

            # 可视化 ROI 框
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

        # 3. 绘制路径细节
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

        # 4. 发送数据到 MuJoCo
        if self.sim:
            scan_h = self.get_float(self.inp_height, 0.1)
            offset = np.array(SAMPLE_OFFSET)
            pts_m = points / 1000.0 + offset

            self.sim.set_path(pts_m, normals, height=scan_h)
            self.sim.paused = True

            # 激活主窗口焦点
            self.activateWindow()
            self.setFocus()

    def toggle_pause(self):
        if self.sim:
            self.sim.paused = not self.sim.paused

    def keyPressEvent(self, event):
        if not self.sim: return
        if event.key() == Qt.Key_Left:
            self.sim.manual_adjust(-1)
        elif event.key() == Qt.Key_Right:
            self.sim.manual_adjust(1)
        elif event.key() == Qt.Key_Space:
            self.toggle_pause()

    def game_loop(self):

        # 主循环 驱动仿真、渲染画面、更新UI
        if not self.sim: return

        # 1. 物理步进
        self.sim.step()

        # 2. GUI 渲染跳帧优化 (大幅降低 GPU 压力)
        self.render_skip_counter += 1
        if self.render_skip_counter % 2 != 0:
            return

        try:
            # 执行离屏渲染
            img_robot, img_global = self.sim.render_offscreen()
            self.update_camera_views(img_robot, img_global)

            # 更新 FPS 计数
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter * 2  # 补偿跳帧显示的 FPS 计数
                self.fps_counter = 0
                self.last_fps_time = time.time()

            txt = f"FPS: {self.current_fps}\n"
            txt += f"Status: {'PAUSED' if self.sim.paused else 'RUNNING'}"
            self.lbl_stats.setText(txt)

            # 更新进度信息
            idx = self.sim.current_idx
            total = len(self.sim.path_points) if self.sim.path_points is not None else 0
            self.lbl_progress.setText(f"Point: {idx + 1} / {total}")

            # 在样品视图中更新当前点高亮
            if len(self.current_points) > idx:
                current_pos = self.current_points[idx]
                self.plotter.add_mesh(
                    pv.Sphere(radius=self.step_size * 0.4, center=current_pos),
                    color="red", name="Highlight", render_points_as_spheres=True,
                    reset_camera=False
                )
        except Exception:
            pass

    def update_camera_views(self, img_robot, img_global):

        # 将 numpy 图像数据转换为 QPixmap 并显示
        def np2pixmap(img):
            h, w, c = img.shape
            bytes_per_line = 3 * w
            qimg = QImage(img.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)

        self.lbl_cam_robot.setPixmap(np2pixmap(img_robot))
        self.lbl_cam_global.setPixmap(np2pixmap(img_global))

    def closeEvent(self, event):
        # 窗口关闭时清理资源
        self.timer.stop()
        if self.sim:
            self.sim.close()
        # 强制退出，确保杀掉所有子线程
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