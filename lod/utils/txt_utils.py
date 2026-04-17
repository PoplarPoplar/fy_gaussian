# 2024/9/7
#
# 将一些中间结果输出为 txt 的函数

from math import radians
import os
import numpy as np

from utils.aabb_utils import ori_AABB_info

def save_gps_info(file_save_path, lon, lat, altitude):
    """
    保存gps信息
    save_path: str, 保存路径
    lon, lat, altitude: float, 经纬高度
    """
    with open(file_save_path, 'w') as file:
        file.write(f"lon(degree): {lon}\n")
        file.write(f"lat(degree): {lat}\n")
        file.write(f"altitude: {altitude}\n")
        file.write(f"lon(radius): {radians(lon)}\n")
        file.write(f"lat(radius): {radians(lat)}\n")
    print(f"gps info is written to {file_save_path}")
    
def save_tile_info(file_save_path, tiles_list):
    """
    保存划分的 tiles 的信息
    file_save_path: str, 存储路径
    tiles_list: list, 由 list 组成，每个 list 包含 point_min 和 point_max, 均为 np.array(3)
    """
    with open(file_save_path, 'w') as f:
        for i, aabb in enumerate(tiles_list):
            point_min = aabb[0]
            point_max = aabb[1]

            point_min_str = ', '.join(map(str, point_min))
            point_max_str = ', '.join(map(str, point_max))

            f.write(f"tile{i}: {point_min_str}, {point_max_str}\n")
    print(f"LOD tiles info is written to {file_save_path}")

def save_ori_aabb_info(file_save_path, ori_aabbs_info):
    """
    保存原始 AABB 信息到文件
    file_save_path: str, 存储路径
    ori_aabbs_info: list, 每个元素是 ori_AABB_info 对象
    """
    with open(file_save_path, 'w') as f:
        for aabb_info in ori_aabbs_info:
            uid = aabb_info.uid
            center_str = ' '.join(map(str, aabb_info.center))
            axis_str = ' '.join(map(str, aabb_info.axis))

            f.write(f"{uid}: center({center_str}), axis({axis_str})\n")
    print(f"Original AABB info is written to {file_save_path}")
    
def load_ori_aabb_info(file_path):
    """
    从文件加载原始 AABB 信息
    file_path: str, 文件路径
    """
    ori_aabbs_info = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            uid_part, rest = line.split(': ', 1)
            uid = uid_part.strip()
            center_part, axis_part = rest.split(', axis(', 1)
            center_part = center_part.replace('center(', '').strip(')')
            axis_part = axis_part.strip(')')
            center = np.array(list(map(float, center_part.split())))
            axis = np.array(list(map(float, axis_part.split())))
            
            ori_aabb_info = ori_AABB_info(uid=uid, center=center, axis=axis)
            ori_aabbs_info.append(ori_aabb_info)
    
    return ori_aabbs_info

def save_nodes_info(file_save_path, node_nums):
    """
    记录最终各个 Tile 的节点数以及节点总数
    file_save_path: str, 存储路径
    node_nums: dict, 存储每个 tile 的节点数
    """
    total_node_num = sum(node_nums.values())
    with open(file_save_path, 'w') as file:
        file.write(f"Total node_num: {total_node_num}\n")
        for tile, num in node_nums.items():
            file.write(f"{tile}: {num}\n")
        print(f"Nodes info is written to {file_save_path}")

def save_pyramid_levels_to_txt(file_save_path, pyramid_to_level):
    """
    将 pyramid_to_level 字典中的键值对按顺序保存到 txt 文件中
    file_save_path: str, 保存的文件名
    pyramid_to_level: dict, 存储每个层级和对应金字塔训练结果的字典
    """
    with open(file_save_path, 'w') as f:
        for lod_level, pyramid_level in pyramid_to_level.items():
            f.write(f"Level {lod_level}: {pyramid_level}\n")
    
    print(f"{file_save_path} have been written !")
    
def save_pyramid_ori_aabb_info_txt(output_path, ori_aabbs_info_dic):
    """
    将金字塔各层级原始 AABB 信息输出到 txt 文件
    output_path: str, 输出路径
    ori_aabbs_info_dic: dict, 存储金字塔各层级原始 AABB 信息
    """
    for pyramid_level, ori_aabbs_info in ori_aabbs_info_dic.items():
        save_path = os.path.join(output_path, f"{pyramid_level}_ori_AABB_info.txt")
        save_ori_aabb_info(save_path, ori_aabbs_info)