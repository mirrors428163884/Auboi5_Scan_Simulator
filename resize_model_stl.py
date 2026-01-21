import trimesh
import numpy as np
import os
import math

INPUT_FILE = 'models/side_holder.stl'
OUTPUT_FILE = 'models/side_holder_output.stl'

# 1. 缩放因子
# 0.001 表示将毫米转换为米 (1/1000)，1.0 表示不缩放
SCALE_FACTOR = 0.001

# 2. 旋转角度 (单位：度)
# Roll  (翻滚): 绕 X 轴旋转
# Pitch (俯仰): 绕 Y 轴旋转
# Yaw   (偏航): 绕 Z 轴旋转 (原有的旋转)
# 正数通常表示逆时针，负数表示顺时针
ROTATE_ROLL_DEG = 0.0  # 绕 X 轴
ROTATE_PITCH_DEG = 0.0  # 绕 Y 轴
ROTATE_YAW_DEG = 0.0  # 绕 Z 轴


def get_rotation_matrix(axis, degrees):
    """
    根据轴和角度生成 4x4 旋转矩阵
    axis: 'x', 'y', 或 'z'
    degrees: 角度
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.eye(4)

    if axis == 'x':
        # 绕 X 轴旋转
        # [ 1  0   0   0 ]
        # [ 0  c  -s   0 ]
        # [ 0  s   c   0 ]
        # [ 0  0   0   1 ]
        rotation_matrix[1, 1] = c
        rotation_matrix[1, 2] = -s
        rotation_matrix[2, 1] = s
        rotation_matrix[2, 2] = c
    elif axis == 'y':
        # 绕 Y 轴旋转
        # [ c  0   s   0 ]
        # [ 0  1   0   0 ]
        # [-s  0   c   0 ]
        # [ 0  0   0   1 ]
        rotation_matrix[0, 0] = c
        rotation_matrix[0, 2] = s
        rotation_matrix[2, 0] = -s
        rotation_matrix[2, 2] = c
    elif axis == 'z':
        # 绕 Z 轴旋转
        # [ c -s   0   0 ]
        # [ s  c   0   0 ]
        # [ 0  0   1   0 ]
        # [ 0  0   0   1 ]
        rotation_matrix[0, 0] = c
        rotation_matrix[0, 1] = -s
        rotation_matrix[1, 0] = s
        rotation_matrix[1, 1] = c

    return rotation_matrix


def process_stl():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    mesh = trimesh.load(INPUT_FILE)

    # 1: 打印原始尺寸
    extents = mesh.extents
    print(f"\n[原始] 模型尺寸 (X, Y, Z):")
    print(f"  {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f}")

    # 安全检查：防止重复缩小
    if SCALE_FACTOR < 0.1 and np.max(extents) < 5.0:
        print("\n警告: 模型看起来已经是米单位了 (尺寸很小)。")
        print(f"你确定还要执行 {SCALE_FACTOR} 倍的缩放吗？")
        confirm = input("输入 'y' 继续，其他键跳过缩放仅旋转: ")
        if confirm.lower() != 'y':
            print("已跳过缩放操作。")
        else:
            print(f"执行缩放: {SCALE_FACTOR}")
            mesh.apply_scale(SCALE_FACTOR)
    else:
        print(f"执行缩放: {SCALE_FACTOR}")
        mesh.apply_scale(SCALE_FACTOR)

    # 2: 执行旋转 (按顺序应用：Roll -> Pitch -> Yaw)
    # 这个顺序可以根据需要调整，但通常顺序会影响最终结果

    # --- 2.1 Roll (绕 X 轴) ---
    if abs(ROTATE_ROLL_DEG) > 0.001:
        print(f"执行旋转: Roll (绕 X 轴) {ROTATE_ROLL_DEG} 度")
        mat_x = get_rotation_matrix('x', ROTATE_ROLL_DEG)
        mesh.apply_transform(mat_x)

    # --- 2.2 Pitch (绕 Y 轴) ---
    if abs(ROTATE_PITCH_DEG) > 0.001:
        print(f"执行旋转: Pitch (绕 Y 轴) {ROTATE_PITCH_DEG} 度")
        mat_y = get_rotation_matrix('y', ROTATE_PITCH_DEG)
        mesh.apply_transform(mat_y)

    # --- 2.3 Yaw (绕 Z 轴) ---
    if abs(ROTATE_YAW_DEG) > 0.001:
        print(f"执行旋转: Yaw (绕 Z 轴) {ROTATE_YAW_DEG} 度")
        mat_z = get_rotation_matrix('z', ROTATE_YAW_DEG)
        mesh.apply_transform(mat_z)

    if abs(ROTATE_ROLL_DEG) < 0.001 and abs(ROTATE_PITCH_DEG) < 0.001 and abs(ROTATE_YAW_DEG) < 0.001:
        print("\n所有旋转角度均为 0，跳过旋转。")

    # 3: 打印最终结果并保存
    new_extents = mesh.extents
    print(f"\n[最终] 模型尺寸 (X, Y, Z):")
    print(f"  {new_extents[0]:.4f} x {new_extents[1]:.4f} x {new_extents[2]:.4f}")

    mesh.export(OUTPUT_FILE)
    print(f"\n成功保存至: {OUTPUT_FILE}")
    print(f"包含操作: 缩放 x{SCALE_FACTOR}, Roll={ROTATE_ROLL_DEG}, Pitch={ROTATE_PITCH_DEG}, Yaw={ROTATE_YAW_DEG}")


if __name__ == '__main__':
    process_stl()