# 2024/8/31
#
# 用于处理 AABB 的函数
import os
import numpy as np
from typing import NamedTuple

def read_bounds_file(bounds_file):
    """
    读取并返回 chunk 的 AABB 包围盒的中心以及轴长
    bounds_file: str, center.txt 以及 extent.txt 的存储地址
    """
    with open(os.path.join(bounds_file, "center.txt")) as centerfile:
        with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
            centerline = centerfile.readline()
            extentline = extentfile.readline()

            c = centerline.split(' ')
            e = extentline.split(' ')

    c = [float(c[0]), float(c[1]), float(c[2])]
    # TODO 是否需要这个 min 函数， 目前是金字塔的 axis 过大
    e = [float(e[0]), float(e[1]), min(float(e[2]), 10000)]

    return np.array(c), np.array(e)

class ori_AABB_info(NamedTuple):
    uid: str
    center: np.array
    axis: np.array

def calculate_scene_aabb(ori_aabbs_info):
    """
    根据原始的 AABB 信息，计算整个场景的 AABB 信息
    ori_aabbs_info: list, 每个元素是 ori_AABB_info, 包含中心和轴长
    """
    centers = np.array([aabb_info.center for aabb_info in ori_aabbs_info])
    axes = np.array([aabb_info.axis for aabb_info in ori_aabbs_info])

    min_points = centers - axes / 2
    max_points = centers + axes / 2

    overall_min_point = np.min(min_points, axis=0)
    overall_max_point = np.max(max_points, axis=0)
    
    return overall_min_point, overall_max_point

def is_intersecting(aabb1, aabb2):
    """
    判断两个 AABB 是否相交。
    aabb1: tuple, (min_point, max_point) 第一个 AABB 的最小点和最大点
    aabb2: tuple, (min_point, max_point) 第二个 AABB 的最小点和最大点
    """
    min1, max1 = aabb1
    min2, max2 = aabb2
    return np.all(max1 >= min2) and np.all(min1 <= max2)

def find_intersecting_aabbs(target_aabb, ori_aabbs_info):
    """
    找出与目标 AABB 相交的 AABB 列表。
    target_aabb: tuple, (min_point, max_point) 目标 AABB 的最小点和最大点
    ori_aabbs_info: list, 每个元素是 ori_AABB_info 对象
    """
    intersecting_aabbs = []

    for aabb_info in ori_aabbs_info:
        min_point = aabb_info.center - aabb_info.axis / 2
        max_point = aabb_info.center + aabb_info.axis / 2
        aabb = (min_point, max_point)
        if is_intersecting(target_aabb, aabb):
            intersecting_aabbs.append(aabb_info)

    return intersecting_aabbs

class getAABB:
    """
    获取 AABB 包围盒的信息以及对 AABB 进行操作
    """
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point
        self.get_information()

    def get_information(self):
        """
        获得包围盒的中心、半轴长、体积
        """
        self.center = (self.max_point + self.min_point) / 2
        self.half_dim = (self.max_point - self.min_point) / 2
        self.volumn = 8 * self.half_dim[0] * self.half_dim[1] * self.half_dim[2]

    def get_boundingVolume_box(self):
        bounding_box = [self.center[0], self.center[1], self.center[2],
                        self.half_dim[0], 0, 0,
                        0, self.half_dim[1], 0,
                        0, 0, self.half_dim[2]]
        return bounding_box

    def get_radius(self):
        return np.linalg.norm(self.half_dim)
    
    def get_radius_withoutz(self):
        half_dim_xy = self.half_dim[:2]
        return np.linalg.norm(half_dim_xy)