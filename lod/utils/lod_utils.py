# 2024/08/31
#
# 从最粗糙层至最精细层创建 LOD 树

import os
import numpy as np
from plyfile import PlyData, PlyElement

from utils.aabb_utils import find_intersecting_aabbs
from utils.ply_utils import split_ply

def calculate_AABB_from_points(gs_ply):
    """
    根据高斯椭球坐标计算包围盒大小
    # TODO 考虑高斯椭球 scale 的均值
    gs_ply: Plydata['vertex']
    """
    points = np.vstack([gs_ply['x'], gs_ply['y'], gs_ply['z']]).T
    new_min_point = np.min(points, axis=0)
    new_max_point = np.max(points, axis=0)
    return new_max_point, new_min_point

def split_aabb(min_point, max_point, d, dim_index):
    """
    在第 dim_index 维度上，将 AABB 以 d 为边界进行划分
    min_point, max_point: np.array, 分别表示 AABB 的最小点和最大点
    d: float, 分割的坐标
    dim_index: int, 分割的轴, 0、1、2分别代表 x、y、z 轴
    """
    
    if d >= min_point[dim_index] and d < max_point[dim_index]:
        # 复制原始 AABB
        child0_min_point = np.copy(min_point)
        child0_max_point = np.copy(max_point)
        child1_min_point = np.copy(min_point)
        child1_max_point = np.copy(max_point)
        
        # 划分 AABB
        child0_max_point[dim_index] = d
        child1_min_point[dim_index] = d
        
        # 创建两个新的 AABB
        child0 = [child0_min_point, child0_max_point]
        child1 = [child1_min_point, child1_max_point]
        
        return [child0, child1]
    else:
        assert False, "The current point is not within the segmentation range!"

class create_LOD:
    def __init__(self, min_point, max_point, name, save_path, level, aabbs_info, plyfile=[]):
    # def __init__(self, min_point, max_point, name, save_path, level, aabbs_info):
        self.min_point = min_point
        self.max_point = max_point
        self.name = name
        self.save_path = save_path
        self.level = level
        self.aabbs_info = aabbs_info
        self.child = {}
        self.method = 0 # 数据划分方式，0代表自适应四分，目前只实现了自适应四分
        self.plyfile = plyfile
    
    def get_child(self, plyfile_path, total_level, max_splat_num, is_child=True, is_pyramid = False, ori_aabb_info = []):
        """
        获得当前节点 AABB 的子节点
        plyfile_path: str, 各个 tile 训练结果路径
        total_level: int, LOD 总层数
        """
        if self.method == 0:
            ply = self.get_ply(plyfile_path, is_child, is_pyramid, ori_aabb_info)
            if ply == []:
                child_aabb_list = []
            else:
                child_aabb_list = self.SplitLODNode_axi(ply, max_splat_num)
        else:
            assert False, "AABB partitioning method currently not supported!"
        
        level = self.level + 1
        for i in range(len(child_aabb_list)):
            child_aabb = child_aabb_list[i]
            min_point, max_point = child_aabb[0], child_aabb[1]
            # if level == total_level - 1:
            #     save_path = self.save_path
            # else:
            #     save_path = os.path.join(self.save_path, f"{i}")
            save_path = self.save_path
            ori_aabbs_info = ori_aabb_info if is_pyramid else self.aabbs_info
            aabbs_info = find_intersecting_aabbs((min_point, max_point), ori_aabbs_info)
            sub_ply = split_ply(min_point, max_point, ply)
            num_points = len(sub_ply["vertex"])

            # 修改名称属性
            name_parts = self.name.split('_')
            level_part = name_parts[-2]
            new_level = int(level_part[1:]) + 1
            name_parts[-2] = 'L' + str(new_level)
            new_filename = '_'.join(name_parts)
            new_name = new_filename + str(i)
            # 创建子节点
            # child_lodtree = create_LOD(min_point, max_point, new_name, save_path, level, aabbs_info)
            if num_points != 0:
                child_lodtree = create_LOD(min_point, max_point, new_name, save_path, level, aabbs_info, sub_ply)
                self.child[i] = child_lodtree
    
    def get_ply(self, plyfile_path, is_child=False, is_pyramid = False, ori_aabb_info = []):
        """
        生成当前节点 AABB 块的 ply 数据
        plyfile_path: str, 各个 tile 训练结果路径
        is_child: bool, False 代表利用当前层级的 ply 数据生成，用于成果保存
                        True 代表下一层级的 ply 数据生成，用于 AABB 划分
                        默认为 False
        """
        vertices = []
        if is_pyramid:
            aabbs_info = find_intersecting_aabbs((self.min_point, self.max_point), ori_aabb_info)
            if aabbs_info == []:
                return []
        else:
            aabbs_info = self.aabbs_info
        for aabb_info in aabbs_info:
            if is_child:
                plydata_path = os.path.join(plyfile_path, aabb_info.uid, "lod_temp", f"point_cloud_{self.level + 1}.ply")
            else:
                plydata_path = os.path.join(plyfile_path, aabb_info.uid, "lod_temp", f"point_cloud_{self.level}.ply")
            ply_data = PlyData.read(plydata_path)
            ply_data = split_ply(self.min_point, self.max_point, ply_data)
            vertices.append(ply_data)
        
        ply_data = vertices[0]
        l = len(vertices) - 1
        for i in range(l):
            ply_data_add = vertices[i + 1]
            merged_vertices = np.concatenate((ply_data['vertex'].data, ply_data_add['vertex'].data))
            ply_data = PlyData([PlyElement.describe(merged_vertices, 'vertex')])
        return ply_data
    
    def SplitNode_kd4(self, plydata):
        """
        根据 plydata 对当前节点 AABB 进行四分
        plydata: Plydata, 点云数据 
        """
        child_aabb = []
        extents = self.max_point - self.min_point
        dim = 0 if extents[0] >= extents[1] else 1
        
        # 计算投影到当前轴的中位数   
        vertex_data = plydata['vertex']
        x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
        points = np.array([x, y, z]).T
        split_values = points[:, dim]
        median = np.median(split_values)
        # 根据中位数划分当前轴
        aabb_list = split_aabb(self.min_point, self.max_point, median, dim)
        # 对所获得的子 AABB 再度划分以实现四分
        for i, aabb in enumerate(aabb_list):
            extents = aabb[1] - aabb[0]
            dim = 0 if extents[0] >= extents[1] else 1
            child_ply = split_ply(aabb[0], aabb[1], plydata)
            vertex_data = child_ply['vertex']
            x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
            points = np.array([x, y, z]).T
            split_values = points[:, dim]
            median = np.median(split_values)
            child_aabb_list = split_aabb(aabb[0], aabb[1], median, dim)
            child_aabb.extend(child_aabb_list)
    
        return child_aabb
    
    def SplitLODNode(self, plydata, max_splat_num):
        def recursive_split(plydata, min_point, max_point, max_splat_num):
            vertex_data = plydata['vertex']
            if len(vertex_data) <= max_splat_num:
                return [[min_point, max_point]]
            # 选择x, y中最长轴
            extents = max_point - min_point
            dim = 0 if extents[0] >= extents[1] else 1
            # dim = np.argmax(extents)
            
            # 找到中位数作为分割点
            x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
            points = np.array([x, y, z]).T
            split_values = points[:, dim]
            median = np.median(split_values)
            
            # 根据中位数划分AABB，得到两个子AABB
            aabb_list = split_aabb(min_point, max_point, median, dim)
            
            result_aabbs = []
            for i, aabb in enumerate(aabb_list):
                # 获取子 AABB 中的点
                child_ply = split_ply(aabb[0], aabb[1], plydata)
                result_aabbs.extend(recursive_split(child_ply, aabb[0], aabb[1], max_splat_num))
            return result_aabbs
        return recursive_split(plydata, self.min_point, self.max_point, max_splat_num)
    
    def SplitLODNode_axi(self, plydata, max_splat_num):
        vertex_data = plydata['vertex']
        num_points = len(vertex_data)

        # 如果点数小于等于max_splat_num，直接返回当前AABB
        if num_points <= max_splat_num:
            return [[self.min_point, self.max_point]]

        # 选择x, y, z中最长轴
        extents = self.max_point - self.min_point
        #dim = 0 if extents[0] >= extents[1] else 1
        dim = np.argmax(extents)

        points = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        split_values = points[:, dim]

        # 计算分块数
        num_blocks = (num_points + max_splat_num - 1) // max_splat_num  # 向上取整确定分块数

        # 按最长轴的分位数进行分割
        percentiles = np.linspace(0, 100, num_blocks + 1)
        split_values = np.percentile(split_values, percentiles)

        result_aabbs = []
        
        # 遍历每个分块，生成相应的AABB
        for i in range(num_blocks):
            # 根据分位数确定子块的范围
            min_split = split_values[i]
            max_split = split_values[i + 1]

            # 创建新的 min_point 和 max_point
            new_min_point = np.copy(self.min_point)
            new_max_point = np.copy(self.max_point)

            # 更新当前维度上的分割范围
            new_min_point[dim] = min_split
            new_max_point[dim] = max_split

            result_aabbs.append([new_min_point, new_max_point])

        return result_aabbs
        
class create_Tiles:
    def __init__(self, min_point, max_point, aabbs_info, method):
        self.min_point = min_point
        self.max_point = max_point
        self.aabbs_info = aabbs_info
        self.method = method # 数据划分方式，0代表自适应四分，目前只实现了自适应四分
    
    def get_tiles(self, plyfile_path, max_splat_num):
        """
        获得当前节点 AABB 的子节点
        plyfile_path: str, 各个 train_tile 的训练结果路径
        max_splat_num: int, 每个块的最大点数
        """
        if self.method == 0:
            ply = self.get_ply(plyfile_path)
            child_aabb_list = self.SplitLODNode(ply, max_splat_num)
        else:
            assert False, "AABB partitioning method currently not supported!"
        return child_aabb_list
    
    def get_ply(self, plyfile_path):
        """
        生成当前节点的 ply 数据
        plyfile_path: str, 各个 tile 训练结果路径
        """
        vertices = []
        for aabb_info in self.aabbs_info:
            plydata_path = os.path.join(plyfile_path, aabb_info.uid, "lod_temp", "point_cloud_0.ply")
            ply_data = PlyData.read(plydata_path)
            ply_data = split_ply(self.min_point, self.max_point, ply_data)
            vertices.append(ply_data)
        
        ply_data = vertices[0]
        l = len(vertices) - 1
        for i in range(l):
            ply_data_add = vertices[i + 1]
            merged_vertices = np.concatenate((ply_data['vertex'].data, ply_data_add['vertex'].data))
            ply_data = PlyData([PlyElement.describe(merged_vertices, 'vertex')])
        return ply_data
    
    def SplitLODNode(self, plydata, max_splat_num):
        def recursive_split(plydata, min_point, max_point, max_splat_num):
            vertex_data = plydata['vertex']
            if len(vertex_data) <= max_splat_num:
                return [[min_point, max_point]]
            # 选择x, y中最长轴
            extents = max_point - min_point
            dim = 0 if extents[0] >= extents[1] else 1
            
            # 找到中位数作为分割点
            x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
            points = np.array([x, y, z]).T
            split_values = points[:, dim]
            median = np.median(split_values)
            
            # 根据中位数划分AABB，得到两个子AABB
            aabb_list = split_aabb(min_point, max_point, median, dim)
            
            result_aabbs = []
            for i, aabb in enumerate(aabb_list):
                # 获取子 AABB 中的点
                child_ply = split_ply(aabb[0], aabb[1], plydata)
                result_aabbs.extend(recursive_split(child_ply, aabb[0], aabb[1], max_splat_num))
            return result_aabbs
        return recursive_split(plydata, self.min_point, self.max_point, max_splat_num)