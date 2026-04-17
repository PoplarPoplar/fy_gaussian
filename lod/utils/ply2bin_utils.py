# 2024/8/23
# 
# 将 ply 文件转换成 bin 文件
# 遵循高斯 content 预处理标准
#
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
import numpy as np

from utils.buffer_write_utils import BufferWriter
from utils.coord_utils import coord

C0 = 0.28209479177387814

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(vec):
    """
    归一化一个四维向量。如果向量长度为0，则返回(0, 0, 0, 1)
    """
    length = np.linalg.norm(vec)  # 计算向量二范数
    
    if length == 0:
        # 如果向量长度为0，返回 (0, 0, 0, 1)
        return np.array([0, 0, 0, 1], dtype=float)
    else:
        # 归一化向量
        return vec / length
   
def build_rotation(rots):
    """
    将四元数转换为旋转矩阵
    """
    norms = np.linalg.norm(rots, axis=1)[:, np.newaxis]
    qs = rots / norms

    r = qs[:, 0]
    x = qs[:, 1]
    y = qs[:, 2]
    z = qs[:, 3]

    R = np.zeros((rots.shape[0], 3, 3))

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R

def computeCov3d(R_matrices, scales):
    '''
    得到协方差矩阵上三角矩阵
    '''
    R_scaled = R_matrices * scales[:, np.newaxis, :]  # 形状变为 (N, 3, 3)
    
    m0 = R_scaled[:, 0, :]
    m1 = R_scaled[:, 1, :]
    m2 = R_scaled[:, 2, :]

    # 计算协方差矩阵 covA 和 covB
    covA = np.zeros((R_matrices.shape[0], 3))  # 形状为 (N, 3)
    covB = np.zeros((R_matrices.shape[0], 3))  # 形状为 (N, 3)

    covA[:, 0] = np.sum(m0 * m0, axis=1)
    covA[:, 1] = np.sum(m0 * m1, axis=1)
    covA[:, 2] = np.sum(m0 * m2, axis=1)

    covB[:, 0] = np.sum(m1 * m1, axis=1)
    covB[:, 1] = np.sum(m1 * m2, axis=1)
    covB[:, 2] = np.sum(m2 * m2, axis=1)

    return covA, covB

def ply_to_bin(ply_data, min_point, max_point, output_path, rot, coord_type='XYZ', scene_scale=1.0, euler_angle=None):
    """
    优化后的 ply_to_bin 函数，使用向量化和批量处理。
    """
    # 旋转和缩放 AABB
    Coord = coord(coord_type, scale=scene_scale, angles=euler_angle)
    max_point, min_point = Coord.transform_box(max_point, min_point)

    # 提取点数据
    plydata = ply_data['vertex']
    num_points = len(plydata)
    points_to_add = (4096 - num_points % 4096) if num_points % 4096 != 0 else 0
    total_write_splat = num_points + points_to_add

    buffer_writer = BufferWriter(total_write_splat)
    buffer_writer.write_header(total_write_splat, min_point, max_point)

    x, y, z = plydata['x'], plydata['y'], plydata['z']
    f_dc_0, f_dc_1, f_dc_2 = plydata['f_dc_0'], plydata['f_dc_1'], plydata['f_dc_2']
    opacity = sigmoid(plydata["opacity"])
    
    scale_0, scale_1, scale_2 = np.exp(plydata['scale_0']), np.exp(plydata['scale_1']), np.exp(plydata['scale_2'])
    rot_0, rot_1, rot_2, rot_3 = plydata['rot_0'], plydata['rot_1'], plydata['rot_2'], plydata['rot_3']

    positions = Coord.transform_points(Coord.scale * np.vstack([x, y, z]).T)
    positions =np.einsum('ijk,ik->ij', rot.transpose(0,2,1), positions)
    scales = Coord.scale * np.vstack([scale_0, scale_1, scale_2]).T
    rots = normalize(np.vstack([rot_0, rot_1, rot_2, rot_3]).T)
    
    R_matrices = build_rotation(rots)
    R_matrices = np.matmul(rot.transpose(0,2,1), R_matrices)
    covA, covB = computeCov3d(R_matrices, scales)
    
    sh_0 = np.array([f_dc_0, f_dc_1, f_dc_2]).T * C0 + 0.5
    colors = np.clip(255 * np.hstack((sh_0, opacity[:, np.newaxis])), 0, 255).astype(np.uint8)  # 限定范围并转为整数

    buffer_writer.write_points(positions, covA, covB, colors)

    # 处理多余点
    if points_to_add:
        positions = np.random.uniform(min_point, max_point, size=(points_to_add, 3))
        cov_as = np.zeros((points_to_add, 3), dtype=np.float32)
        cov_bs = np.zeros((points_to_add, 3), dtype=np.float32)
        colors = np.zeros((points_to_add, 4), dtype=np.uint8)
        buffer_writer.write_points(positions, cov_as, cov_bs, colors)

    # 保存文件
    with open(output_path, 'wb') as f:
        f.write(buffer_writer.get_buffer())

    return min_point, max_point