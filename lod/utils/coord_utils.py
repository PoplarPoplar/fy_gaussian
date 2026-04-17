# 2024/08/27
#
# 用于处理坐标轴的旋转、缩放与平移

import math
import numpy as np
from pyproj import Proj, CRS, Transformer

DEG_TO_RAD = math.pi / 180

def generate_enu_matrix(lat, lon):
    """
    根据给定的经纬度生成 East-North-up(ENU) 转换矩阵
    lat: float, 纬度（角度制）
    lon: float, 经度（角度制）
    """
    # 转为弧度制
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # 计算各方向单位向量并创建 enu 矩阵
    north = np.array([-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)])
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
    up = np.array([np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)])
    enu_matrix = np.vstack([east, north, up])

    return enu_matrix

# def coord_transform(vector):
#     """
#     将点的坐标从 CGCS2000 / 3-degree Gauss-Kruger CM 114E (EPSG:4547) 转换到 WGS 84 (EPSG:4326)
#     vector: np.array[3], 包含三个坐标值 [x, y, z]
#     """
#     # 创建一个转换器从 EPSG:4547 到 EPSG:4326
#     source_proj = Proj(proj='tmerc', lat_0=0, lon_0=114, k=1, x_0=500000, y_0=0, ellps='GRS80', datum='WGS84')
#     # WGS 84
#     target_proj = Proj(proj='longlat', datum='WGS84')
#     transformer = Transformer.from_proj(source_proj, target_proj, always_xy=True)
#     lon, lat = transformer.transform(vector[0], vector[1])
#     altitude = vector[2]
#     return lon, lat, altitude

def coord_transform(vector, source_epsg):
    """
    将点的坐标从指定的EPSG坐标系转换到 WGS 84 (EPSG:4326)
    vector: np.array[3], 包含三个坐标值 [x, y, z]
    source_epsg: int, 源坐标系的EPSG编号
    """
    # 创建转换器从指定 EPSG 坐标系到 EPSG:4326
    source_crs = CRS.from_epsg(source_epsg)
    target_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    lon, lat = transformer.transform(vector[0], vector[1])
    altitude = vector[2]
    return lon, lat, altitude

def create_transform(lon, lat, altitude, R_transpose):
    """
    生成 transform 矩阵
    lon, lat, altitude: float, 经纬高度
    R_transpose: 由四元数生成的旋转矩阵
    """
    transformer_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    ecef_x, ecef_y, ecef_z = transformer_to_ecef.transform(lon, lat, altitude)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = np.matmul(R_transpose.squeeze(), generate_enu_matrix(lat, lon)).T
    transform_matrix[:3, 3] = [ecef_x, ecef_y, ecef_z]
    return transform_matrix

def rotation_from_euler_angles(angles):
    """
    通过输入的欧拉角（角度制）计算旋转矩阵
    """
    R = np.zeros((4, 4))
    ex, ey, ez = angles[0], angles[1], angles[2]
    ex *= DEG_TO_RAD
    ey *= DEG_TO_RAD
    ez *= DEG_TO_RAD

    # 计算欧拉角的 sin 和 cos 值
    s1 = math.sin(-ex)
    c1 = math.cos(-ex)
    s2 = math.sin(-ey)
    c2 = math.cos(-ey)
    s3 = math.sin(-ez)
    c3 = math.cos(-ez)

    # Set rotation matrix elements
    R[0, 0] = c2 * c3
    R[1, 0] = -c2 * s3
    R[2, 0] = s2
    R[3, 0] = 0

    R[0, 1] = c1 * s3 + c3 * s1 * s2
    R[1, 1] = c1 * c3 - s1 * s2 * s3
    R[2, 1] = -c2 * s1
    R[3, 1] = 0

    R[0, 2] = s1 * s3 - c1 * c3 * s2
    R[1, 2] = c3 * s1 + c1 * s2 * s3
    R[2, 2] = c1 * c2
    R[3, 2] = 0

    R[0, 3] = 0
    R[1, 3] = 0
    R[2, 3] = 0
    R[3, 3] = 1

    return R

def quaternion_from_matric(matric):
    """
    从旋转矩阵得到四元数
    """
    # 将 matric 展平后提取数据
    d = matric.flatten()
    m00 = d[0]
    m01 = d[4]
    m02 = d[8]
    m10 = d[1]
    m11 = d[5]
    m12 = d[9]
    m20 = d[2]
    m21 = d[6]
    m22 = d[10]
    
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.array([0, 0, 0, 1])
        return v / norm

    v0 = normalize(np.array([m00, m01, m02]))
    if np.array_equal(v0, [0, 0, 0, 1]):
        return v0
    v1 = normalize(np.array([m10, m11, m12]))
    if np.array_equal(v1, [0, 0, 0, 1]):
        return v1
    v2 = normalize(np.array([m20, m21, m22]))
    if np.array_equal(v2, [0, 0, 0, 1]):
        return v2

    m00, m01, m02 = v0
    m10, m11, m12 = v1
    m20, m21, m22 = v2

    if m22 < 0:
        if m00 > m11:
            quat = np.array([1 + m00 - m11 - m22, m01 + m10, m20 + m02, m12 - m21])
        else:
            quat = np.array([m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m20 - m02])
    else:
        if m00 < -m11:
            quat = np.array([m20 + m02, m12 + m21, 1 - m00 - m11 + m22, m01 - m10])
        else:
            quat = np.array([m12 - m21, m20 - m02, m01 - m10, 1 + m00 + m11 + m22])

    # Normalize quaternion
    quat /= np.linalg.norm(quat)

    return quat

class coord:
    def __init__(self, coord_type, scale=1, angles=None):
        self.type = coord_type
        self.get_rotation_matric_and_quaternion(angles)
        self.scale = scale
        
    def get_rotation_matric_and_quaternion(self, angles):
        if angles is not None:
            self.m = rotation_from_euler_angles(angles)
        else:
            if self.type == "XYZ":
                self.m = np.identity(4)
            if self.type == "XNZY":
                self.m = rotation_from_euler_angles([90, 0, 0])
            if self.type == "XZY":
                self.m = rotation_from_euler_angles([-90, 0, 0])
            if self.type == "XYNZ":
                self.m = rotation_from_euler_angles([180, 0, 180])
        
        self.q = quaternion_from_matric(self.m)
        
    def transform_point(self, vec):
        """
        使用变换矩阵变换一个 3D 点
        """
        # 将点转换为4D齐次坐标 (x, y, z, 1)
        vec_homogeneous = np.append(vec, 1)
        transformed_vec = self.m @ vec_homogeneous
        return transformed_vec[:3]
    
    def transform_points(self, vecs):
        """
        使用变换矩阵批量变换一组 3D 点
        vecs: n x 3 的点云数据
        """
        # 将 3D 点转换为 4D 齐次坐标 (x, y, z, 1)
        vecs_homogeneous = np.hstack((vecs, np.ones((vecs.shape[0], 1))))
        
        # 批量应用变换矩阵并移除最后一列齐次坐标
        transformed_vecs = (self.m @ vecs_homogeneous.T).T
        
        # 只返回前三维 (x, y, z) 并进行缩放
        return transformed_vecs[:, :3] * self.scale
    
    def quaternion_multiply(self, r):
        """
        计算两个四元数的乘积
        """    
        x1, y1, z1, w1 = self.q
        x2, y2, z2, w2 = r
        
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        
        return np.array([x, y, z, w])
    
    def transform_box(self, max_point, min_point):
        """
        使用变换矩阵变换包围盒的最大点和最小点
        返回新的包围盒最大点和最小点
        """
        # 将包围盒的八个角点转换为4D齐次坐标
        corners = np.array([[min_point[0], min_point[1], min_point[2], 1],
                            [max_point[0], min_point[1], min_point[2], 1],
                            [min_point[0], max_point[1], min_point[2], 1],
                            [max_point[0], max_point[1], min_point[2], 1],
                            [min_point[0], min_point[1], max_point[2], 1],
                            [max_point[0], min_point[1], max_point[2], 1],
                            [min_point[0], max_point[1], max_point[2], 1],
                            [max_point[0], max_point[1], max_point[2], 1]])
        
        # 对八个角点进行批量变换
        transformed_corners = (self.m @ corners.T).T[:, :3] * self.scale

        # 计算新的包围盒的最小值和最大值
        new_min = np.min(transformed_corners, axis=0)
        new_max = np.max(transformed_corners, axis=0)

        # 进行变换
        # transformed_corners = (self.m @ corners.T).T
        
        # 提取变换后的坐标
        # transformed_corners = transformed_corners[:, :3]
        # transformed_corners *= self.scale
        
        # 计算新的包围盒的最大值和最小值
        # new_min = np.min(transformed_corners, axis=0)
        # new_max = np.max(transformed_corners, axis=0)
        
        return new_max, new_min
        
if __name__ == "__main__":
    c = coord("XZY")
    print(c.m)
    print(c.q)