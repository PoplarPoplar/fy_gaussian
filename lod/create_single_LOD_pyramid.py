# 2024/9/13
#
# 创建单个 lod_tile 的 3DGS 模型的 LOD

from argparse import ArgumentParser
import os
import numpy as np

from utils.aabb_utils import find_intersecting_aabbs
from utils.lod_utils import create_LOD, calculate_AABB_from_points
from utils.ply2bin_utils import ply_to_bin
from utils.json_utils import create_tile_jsoninfo, create_tile_json

def create_single_LOD_pyramid(tile_id, min_point, max_point, pyramid_to_level, input_path, output_path, 
                              ori_aabbs_info_dic, level, rot, max_splat_num, geo_list):
    """
    创建单个 lod_tile 的 LOD 并保存成果
    tile_id : int, 需要创建的 tile 的 id
    min_point, max_point: np.array, AABB 的最小点和最大点
    pyramid_to_level: dict, 金字塔与 LOD 深度的映射字典
    input_path: str, 3DGS 训练成果保存路径
    output_path: str, 成果保存路径
    ori_aabbs_info_dic: dict, 存储每个金字塔层级的原始 AABB 信息
    level: int, LOD 总层级
    rot: 由四元数生成的旋转矩阵
    max_splat_num: int, 单个块最大点数
    geo_list: list, 几何误差list
    """
    tile_name = f"tile{tile_id}"
    save_path = os.path.join(output_path, tile_name)
    tree_depth = 0
    crop_name = tile_name + '_L13_0'
    ori_aabbs_info = ori_aabbs_info_dic[pyramid_to_level[tree_depth]]
    tile_aabbs_info = find_intersecting_aabbs((min_point, max_point), ori_aabbs_info)
    
    lodtree = create_LOD(min_point, max_point, crop_name, save_path, tree_depth, tile_aabbs_info)

    # 深度优先构建 LOD 树
    print(f"Creating LOD of {tile_name}...")
    node_num = 0
    stack = [(lodtree, tree_depth)]
    while stack:
        node, current_level = stack.pop()
        pyramid_level = pyramid_to_level[current_level]
        # ply_path = os.path.join(input_path, pyramid_level)
        # ply = node.get_ply(ply_path)
        if current_level == 0:
            ply_path = os.path.join(input_path, pyramid_level)
            ply = node.get_ply(ply_path)
        else:
            ply = node.plyfile
        node.max_point, node.min_point = calculate_AABB_from_points(ply['vertex'])
        node_num += 1
        os.makedirs(node.save_path, exist_ok=True)
        # TODO ply 暂时保存到另一个文件夹，方便 debug
        # ply.write(os.path.join(node.save_path, f"{node.name}.ply"))
        if current_level < level - 1:
            pyramid_level = pyramid_to_level[current_level + 1]
            ply_path = os.path.join(input_path, pyramid_level)
            ori_aabbs_info = ori_aabbs_info_dic[pyramid_to_level[current_level+1]]
            node.get_child(ply_path, level, max_splat_num, is_pyramid=True, ori_aabb_info=ori_aabbs_info)  # 生成非叶子节点的子节点
            for child_node in node.child.values():
                stack.append((child_node, current_level + 1))
        # 将 .ply 文件转为 .bin 文件
        bin_path = os.path.join(node.save_path, f"{node.name}.bin")
        node.min_point, node.max_point = ply_to_bin(ply, node.min_point, node.max_point, bin_path, rot)
                
    print(f"LOD of {tile_name} finished !")
    json_info = create_tile_jsoninfo(output_path, lodtree, level, geo_list[:-1])
    return json_info, node_num