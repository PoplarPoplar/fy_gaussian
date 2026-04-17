# 2024/8/31
# 
# 用于处理 ply 的函数

import numpy as np
from plyfile import PlyData, PlyElement
# from shapely.geometry import Point, Polygon

import torch
from simple_knn._C import distCUDA2

def ray_intersect_polygon(points, polygon):
    """
    使用射线法判断多个点是否在多边形内
    points: (N, 2) 点云坐标数组
    polygon: (M, 2) 多边形顶点数组
    返回:
    is_inside: (N,) 布尔数组，True表示点在多边形内部，False表示点在外部
    """
    polygon = np.vstack([polygon, polygon[0]]) # 确保多边形是闭环（首尾相连）

    # 获取多边形边的起始点和终点以及点云的坐标
    p1 = polygon[:-1]
    p2 = polygon[1:]
    x_points = points[:, 0]
    y_points = points[:, 1]

    # 计算每个点与每条边的交点，如果交点数目是奇数，表示点在多边形内部；偶数则在外部
    y_min = np.minimum(p1[:, 1], p2[:, 1])
    y_max = np.maximum(p1[:, 1], p2[:, 1])
    y_check = (y_points[:, None] > y_min) & (y_points[:, None] <= y_max)
    
    denom = p2[:, 0] - p1[:, 0]
    x_intersection = (y_points[:, None] - p1[:, 1]) * (p2[:, 0] - p1[:, 0]) / (p2[:, 1] - p1[:, 1]) + p1[:, 0]
    x_check = x_intersection > x_points[:, None]

    intersect_check = y_check & x_check
    intersections = np.sum(intersect_check, axis=1)
    is_inside = (intersections % 2 == 1)

    return is_inside

def split_ply(min_point, max_point, ply, polygon=np.array([])):
    """
    根据包围盒裁切点云文件，并返回裁切结果。
    min_point: list, AABB 包围盒最小点的坐标
    extent: list, AABB 包围盒最大点的坐标
    ply: PlyData, 待裁切的点云
    polygon: polygon, 用于裁切点云的多边形
    """
    # TODO 裁切是否需要对包围盒进行微小扩大以确保有重叠度
    xmin, ymin, zmin = min_point
    xmax, ymax, zmax = max_point

    vertices = ply['vertex']
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])

    in_bounds = np.logical_and.reduce((x >= xmin, x <= xmax, 
                                       y >= ymin, y <= ymax, 
                                       z >= zmin, z <= zmax))
    
    filter_vertices = vertices[in_bounds]
    
    # if polygon != None:
    if polygon.size != 0:
        # points = [Point(filter_vertices['x'][i], filter_vertices['y'][i]) for i in range(len(filter_vertices))]
        points = np.column_stack((filter_vertices['x'], filter_vertices['y']))
        
        # 仅保留在多边形内的点
        # in_polygon = np.array([polygon.contains(p) for p in points])
        in_polygon = ray_intersect_polygon(points, polygon)
        filter_vertices = filter_vertices[in_polygon]
    return PlyData([PlyElement.describe(filter_vertices, 'vertex')])
    
def downsample_ply(ply, ratio, method):
    """
    采用下采样方法对点云进行按比例采样。
    ply: Plydata, 需要采样的点云
    ratio: float, 采样比例
    method: int, 采样方法, 0代表随机采样
    """
    if method == 0:
        ply = random_sample(ply, ratio)
    else:
        assert False, "downsample method currently not supported!"
    
    return ply

def random_sample(ply, ratio):
    """
    对点云按比例进行随机采样
    ply: Plydata, 需要采样的点云
    ratio: float, 采样比例
    """
    vertex_data = ply['vertex']

    # 计算采样点数并随机采样
    num_points = len(vertex_data)
    num_points_to_sample = int(num_points * ratio)
    indices = np.random.choice(num_points, num_points_to_sample, replace=False)
    sampled_data = vertex_data[indices]
    
    positions = np.vstack([sampled_data['x'], sampled_data['y'], sampled_data['z']]).T
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(positions)).float().cuda()), 0.0000001)
    #scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    scales = torch.log(torch.sqrt(dist2)).cpu().numpy()
    
    # TODO 如何膨胀scale
    # 提取已存在的 scale 属性
    scale_0 = sampled_data['scale_0']
    scale_1 = sampled_data['scale_1']
    scale_2 = sampled_data['scale_2']
    scale_adjustment = np.log(2 * ratio)
    updated_scale_0 = scale_0 - scale_adjustment
    updated_scale_1 = scale_1 - scale_adjustment
    updated_scale_2 = scale_2 - scale_adjustment
    sampled_data['scale_0'] = updated_scale_0
    sampled_data['scale_1'] = updated_scale_1
    sampled_data['scale_2'] = updated_scale_2
    
    # 将更新后的缩放值与计算出的缩放值进行比较，取较小的那个值
    min_scale_0 = np.minimum(updated_scale_0, scales)
    min_scale_1 = np.minimum(updated_scale_1, scales)
    min_scale_2 = np.minimum(updated_scale_2, scales)
    
    # 更新缩放属性
    sampled_data['scale_0'] = min_scale_0
    sampled_data['scale_1'] = min_scale_1
    sampled_data['scale_2'] = min_scale_2
    
    # 创建新的 PlyElement
    vertex_element = PlyElement.describe(sampled_data, 'vertex')
    return PlyData([vertex_element])