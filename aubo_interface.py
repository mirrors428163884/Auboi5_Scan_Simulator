# -*- coding: utf-8 -*-
import time
import logging
import math
import sys


# ==============================================================================
#  Aubo i5 机械臂官方 SDK 接口封装 (基于 pyaubo_sdk)
# ==============================================================================
#
#  【硬件连接指南 / Connection Guide】
#  1. 物理连接:
#     用网线将电脑网口连接到控制柜底部的【Ethernet】接口（严禁连接 EtherCAT 口）。
#
#  2. 电脑 IP 设置 (Windows):
#     - 设置 -> 网络和 Internet -> 以太网 -> IP分配(编辑) -> 手动 -> IPv4
#     - IP地址: 192.168.1.100 (前三段需与机械臂一致，最后一段不同)
#     - 子网掩码: 255.255.255.0
#     - 网关: 留空
#
#  3. 机械臂 IP 确认:
#     - 示教器: 设置 -> 系统 -> 网络。假设为 192.168.1.10
#
#  4. 测试:
#     - 打开 cmd 输入: ping 192.168.1.10
#     - 必须能 ping 通才能运行此代码。
#
#  【使用流程 / Workflow】
#  1. robot = AuboRealRobot("192.168.1.10")
#  2. robot.connect_and_startup()  # 自动完成登录、上电、松刹车
#  3. robot.move_j(start_point)    # 先慢速移动到仿真中的轨迹起始点(防止飞车)
#  4. robot.enter_servo_mode()     # 开启实时透传模式
#  5. while loop:
#         robot.send_servo_point(target_joint) # 50Hz 发送指令
#  6. robot.exit_servo_mode()
#  7. robot.disconnect()
# ==============================================================================


try:
    import pyaubo_sdk
    from pyaubo_sdk import RobotModeType
except ImportError:
    print("【严重错误】未找到 pyaubo_sdk 库！")
    print("请将官方SDK包放入项目目录，或确保已安装 python binding。")
    pyaubo_sdk = None


class AuboRealRobot:
    def __init__(self, ip='192.168.1.100', port=30004):
        self.ip = ip
        self.port = port
        self.client = None
        self.robot_name = None
        self.robot_interface = None
        self.motion_control = None
        self.robot_manage = None
        self.io_control = None

        self.is_connected = False
        self.is_servo_mode = False

        # 配置日志输出
        logging.basicConfig(level=logging.INFO,
                            format='[AUBO] %(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("AuboInterface")

    def connect_and_startup(self):
        """
        全自动初始化：连接 -> 登录 -> 上电 -> 松刹车
        返回: True (成功) / False (失败)
        """
        if pyaubo_sdk is None: return False

        try:
            # 1. 创建客户端并连接
            self.client = pyaubo_sdk.RpcClient()
            self.client.setRequestTimeout(2000)  # 2秒超时

            self.logger.info(f"正在连接机械臂 IP: {self.ip} Port: {self.port} ...")
            self.client.connect(self.ip, self.port)

            if not self.client.hasConnected():
                self.logger.error("连接失败，请检查网线或 IP 设置。")
                return False

            # 2. 登录 (默认账号: aubo, 密码: 123456)
            self.client.login("aubo", "123456")
            if not self.client.hasLogined():
                self.logger.error("登录失败，账号或密码错误。")
                return False

            # 3. 获取核心控制对象
            self.robot_name = self.client.getRobotNames()[0]
            self.robot_interface = self.client.getRobotInterface(self.robot_name)
            self.motion_control = self.robot_interface.getMotionControl()
            self.robot_manage = self.robot_interface.getRobotManage()
            self.io_control = self.robot_interface.getIoControl()

            self.logger.info("连接并登录成功。检查机械臂状态...")

            # 4. 检查状态并自动上电/松刹车
            current_mode = self.robot_interface.getRobotState().getRobotModeType()

            if current_mode == RobotModeType.Running:
                self.logger.info("机械臂已处于 Running 状态，准备就绪。")
            else:
                self.logger.info(f"当前状态: {current_mode}，正在尝试自动上电启动...")

                # 请求上电
                if self.robot_manage.poweron() != 0:
                    self.logger.error("上电请求失败！")
                    return False
                time.sleep(1)  # 等待继电器动作

                # 请求松刹车 (Startup)
                if self.robot_manage.startup() != 0:
                    self.logger.error("松刹车请求失败！")
                    return False

                # 循环等待进入 Running 状态 (最多等 20秒)
                for i in range(20):
                    current_mode = self.robot_interface.getRobotState().getRobotModeType()
                    if current_mode == RobotModeType.Running:
                        self.logger.info("机械臂启动成功！(Brakes Released)")
                        break
                    time.sleep(1)
                    print(f"等待启动中... {i + 1}s")

                if current_mode != RobotModeType.Running:
                    self.logger.error("启动超时，请检查控制柜面板是否有急停按下。")
                    return False

            self.is_connected = True
            return True

        except Exception as e:
            self.logger.error(f"初始化过程中发生异常: {e}")
            return False

    def get_current_joints(self):
        """获取当前关节角度 (rad)，返回 list[float]"""
        if not self.is_connected: return None
        try:
            return list(self.robot_interface.getRobotState().getJointPos())
        except Exception:
            return None

    def move_j(self, joint_angles, acc=0.5, vel=0.5):
        """
        [阻塞式] 关节运动 (MoveJ)
        用途: 在开始扫描任务前，安全地移动到轨迹起始点。
        """
        if not self.is_connected: return False

        # 安全检查: 如果在 servo 模式，必须先退出
        if self.is_servo_mode:
            self.logger.warning("检测到 Servo 模式开启，正在强制退出以执行 MoveJ...")
            self.exit_servo_mode()

        try:
            q = [float(x) for x in joint_angles]
            self.logger.info("正在执行 MoveJ 到目标点...")

            # moveJoint(q, acc, vel, blend_radius, block)
            # acc, vel 单位: rad/s^2, rad/s
            self.motion_control.moveJoint(q, acc, vel, 0.0, 0.0)

            # 阻塞等待到达
            self._wait_arrival()
            self.logger.info("MoveJ 到位。")
            return True
        except Exception as e:
            self.logger.error(f"MoveJ 失败: {e}")
            return False

    def enter_servo_mode(self):
        """开启实时伺服模式 (ServoJ)"""
        if not self.is_connected: return False

        self.logger.info("正在开启 Servo 模式...")
        try:
            self.motion_control.setServoMode(True)
            # 循环检查是否生效
            for _ in range(10):
                if self.motion_control.isServoModeEnabled():
                    self.is_servo_mode = True
                    self.logger.info("Servo 模式已开启 (Ready for streaming)")
                    return True
                time.sleep(0.05)

            self.logger.error("开启 Servo 模式超时！")
            return False
        except Exception as e:
            self.logger.error(f"开启 Servo 模式异常: {e}")
            return False

    def exit_servo_mode(self):
        """退出 Servo 模式"""
        if not self.is_connected: return
        try:
            self.motion_control.setServoMode(False)
            self.is_servo_mode = False
            self.logger.info("Servo 模式已关闭")
        except Exception:
            pass

    def send_servo_point(self, joint_angles, dt=0.02):
        """
        发送单个伺服点 (核心函数)
        Args:
            joint_angles: 6关节角度列表 (rad)
            dt: 期望执行时间 (与控制频率一致，如 50Hz -> 0.02)
        Returns:
            0: 发送成功
            1: 发送失败/异常
            2: 缓冲区已满 (Sim-to-Real 关键: 遇到此返回值需等待)
        """
        if not self.is_connected or not self.is_servo_mode:
            return 1

        try:
            q = tuple(float(x) for x in joint_angles)
            # servoJoint(q, acc, vel, time, lookahead_time, gain)
            # acc=0.0, vel=0.0 表示由控制器自动规划平滑度
            # lookahead_time=0.1, gain=200 是经验参数，适合大多数扫描任务
            ret = self.motion_control.servoJoint(q, 0.0, 0.0, dt, 0.1, 200)

            if ret == 0:
                return 0  # 成功
            elif ret == 2:
                return 2  # 缓冲区满
            else:
                self.logger.warning(f"ServoJ 异常返回值: {ret}")
                return 1
        except Exception as e:
            self.logger.error(f"Servo 发送异常: {e}")
            return 1

    def stop(self):
        """急停"""
        if self.motion_control:
            self.motion_control.stopJoint(2.0)
            self.exit_servo_mode()

    def disconnect(self):
        if self.client:
            self.stop()
            self.client.logout()
            self.client.disconnect()
            self.is_connected = False
            self.logger.info("断开连接")

    def _wait_arrival(self):
        """(内部函数) 阻塞等待运动结束"""
        if not self.motion_control: return
        # 等待运动 ID 变化（开始运动）
        while self.motion_control.getExecId() == -1:
            time.sleep(0.01)
        # 等待运动 ID 变回 -1（运动结束）
        while self.motion_control.getExecId() != -1:
            time.sleep(0.05)


# ==============================================================================
#  简单的连接测试
# ==============================================================================
if __name__ == "__main__":
    print("--- Aubo 接口测试 ---")
    robot = AuboRealRobot("192.168.1.10")  # 请修改为实际 IP

    if robot.connect_and_startup():
        print(f"当前关节角: {robot.get_current_joints()}")
        robot.disconnect()
    else:
        print("连接失败")