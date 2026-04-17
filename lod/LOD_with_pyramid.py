# 2024/9/13
#
# 创建整个重建场景 3DGS 模型的 LOD
# 也可以通过 -t/--tile_list 参数选择要用于创建 LOD 的区块
# 目前只支持 CGCS2000 / 3-degree Gauss-Kruger CM 114E (EPSG:4547) 到 WGS 84 (EPSG:4326) 的计算
# 目前策略为先逐层抽稀，再创建 LOD 树并保存成果

from argparse import ArgumentParser
import os
import numpy as np
from shapely.geometry import Polygon

from split_and_downsample_ply import split_and_downsample_ply_pyramid
from create_single_LOD_pyramid import create_single_LOD_pyramid

from utils.txt_utils import *
from utils.json_utils import read_json_file, create_project_json, create_gps_info
from utils.coord_utils import coord_transform, create_transform
from utils.aabb_utils import ori_AABB_info, calculate_scene_aabb
from utils.ply2bin_utils import build_rotation
from utils.lod_utils import create_Tiles

def calculate_levels(pyramid, down_list, L):
    """
    计算金字塔总 level 以及存储哪个 level 对应的是哪个金字塔训练结果
    pyramid: list, 金字塔训练结果
    done_list: list, 需要抽样的层级
    L: int, 抽样次数
    """
    current_level = 0
    pyramid_to_level = {}

    for scale in pyramid:
        # 如果当前的 scale 在抽样列表中，则增加 L 个抽样层级
        if scale in down_list:
            for _ in range(L):
                pyramid_to_level[current_level] = scale
                current_level += 1
        pyramid_to_level[current_level] = scale
        current_level += 1
    return pyramid_to_level

def get_max_level_for_pyramid(level_mapping, target):
    """
    获得金字塔层级对应的 LOD 树最深节点
    level_mapping: dict, 金字塔与 LOD 深度的映射字典
    target: str, 金字塔层级
    """
    max_level = 0
    
    # 遍历字典，查找目标 scale 对应的层级
    for level, scale in level_mapping.items():
        if scale == target:
            max_level = max(max_level, level)
    
    return max_level
    
def create_LOD_with_pyramid(name, list_pyramid, downsample_list, output_path, args):
    '''
    根据项目地址创建 LODtree 并保存成果。
    name: str, 项目名称
    list_pyramid: list, 用于生成 LOD 的金字塔目录
    downsample_list: list, 需要采样的金字塔层级
    output_path: str, LOD 成果保存路径
    args: 其余输入参数
    '''
    BoundingBox_info_json_path = os.path.join(args.project_dir, "BoundingBox.json")
    if not os.path.exists(BoundingBox_info_json_path):
        assert False, f"There doesn't exist BoundingBox.json in {args.project_dir}!"
    BoundingBox_info = read_json_file(BoundingBox_info_json_path)
    
    # 获取地理信息并生成 transform 矩阵       
    normalizer = np.array([BoundingBox_info["Normalizer"]["origin"]["x"], 
                           BoundingBox_info["Normalizer"]["origin"]["y"],
                           BoundingBox_info["Normalizer"]["origin"]["z"]])
    # 获取坐标轴旋转四元数
    Quaternion = BoundingBox_info.get("Quaternion")
    rot = np.array([Quaternion['w'],Quaternion['x'],Quaternion['y'],Quaternion['z']]) if Quaternion else np.array([1, 0, 0, 0])
    R = build_rotation(rot[np.newaxis,:]) # 变为(1，3，3)
    
    CoordinateSystem_json_path = os.path.join(args.project_dir, "CoordinateSystem.json")
    if not os.path.exists(CoordinateSystem_json_path):
        source_epsg = 4547 # 默认采用 CGCS2000 / 3-degree Gauss-Kruger CM 114E (EPSG:4547) 坐标
    else:
        CoordinateSystem_info = read_json_file(CoordinateSystem_json_path)
        output_coord = CoordinateSystem_info["output"]["value"]
        source_epsg = output_coord.split()[0]
    lon, lat, altitude = coord_transform(normalizer, source_epsg)
    transform_matrix = create_transform(lon, lat, altitude, R)
    create_gps_info(output_path, lon, lat, altitude)
    
    # 获取用户重建 AABB 范围以及 polygon
    user_max_point = np.array([BoundingBox_info['CurrentBoundingBox']['max']['x'],
                               BoundingBox_info['CurrentBoundingBox']['max']['y'],
                               BoundingBox_info['CurrentBoundingBox']['max']['z']])
    user_min_point = np.array([BoundingBox_info['CurrentBoundingBox']['min']['x'],
                               BoundingBox_info['CurrentBoundingBox']['min']['y'],
                               BoundingBox_info['CurrentBoundingBox']['min']['z']])
    GaussianReconParam = BoundingBox_info.get("GaussianReconParam")
    if GaussianReconParam and "polygon" in GaussianReconParam and "vertices" in GaussianReconParam["polygon"]:
        vertices = [(v["x"], v["y"]) for v in GaussianReconParam["polygon"]["vertices"]]
        polygon = Polygon(vertices) if vertices else None
    else:
        polygon = None
    #vertices = [(v["x"], v["y"]) for v in BoundingBox_info["GaussianReconParam"]["polygon"]["vertices"]]
    # if vertices == []:
    #     polygon = None
    # else:
    #     polygon = Polygon(vertices)
    
    # 计算金字塔与 LOD 的映射，便于生成 LOD 成果时找到对应的点云
    pyramid_to_level = calculate_levels(list_pyramid, downsample_list, args.downsample_time)
    pyramid_to_level_save_path = os.path.join(output_path, 'pyramid_to_level.txt')
    save_pyramid_levels_to_txt(pyramid_to_level_save_path, pyramid_to_level)
    lod_level = len(pyramid_to_level)
    
    # 读取原始 AABB 信息并裁切点云
    print("Building AABB trees for pyramid levels...")
    ori_aabbs_info_dic = {}
    for pyramid_level in list_pyramid:
        # 获得金字塔层级对应的 LOD 树最深节点，便于对裁切以及降采样得到的点云命名
        end_level = get_max_level_for_pyramid(pyramid_to_level, pyramid_level)
        is_downsample = pyramid_level in downsample_list
        ori_aabbs_info = []
        input_path = os.path.join(args.project_dir, "outputs", pyramid_level)
        folders = [name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]
        for folder in folders:
            # 按照训练 AABB 以及 polygon 裁切点云并获得训练 AABB 信息以便求交
            center, extent = split_and_downsample_ply_pyramid(input_path, folder, args.downsample_ratio, args.downsample_time, end_level, polygon,
                                                   args.downsample_method, is_downsample)
            ori_aabb_info = ori_AABB_info(uid=folder, center=center, axis=extent)
            ori_aabbs_info.append(ori_aabb_info)
        ori_aabbs_info_dic[pyramid_level] = ori_aabbs_info
    # 保存各个金字塔层级 AABB 信息，用于转换为 czml 文件调试
    save_pyramid_ori_aabb_info_txt(output_path, ori_aabbs_info_dic)
    
    # 获得整个场景的 AABB 中心和轴长
    # TODO 可以考虑直接从 BoundingBox.json 读取 
    #scene_min_point, scene_max_point = calculate_scene_aabb(ori_aabbs_info)
    scene_min_point, scene_max_point = user_min_point, user_max_point
        
    # 将最顶层数据划分为 tiles 并保存划分信息
    print("divide into tiles ...")
    ori_aabbs_info = ori_aabbs_info_dic[list_pyramid[0]]
    scene = create_Tiles(scene_min_point, scene_max_point, ori_aabbs_info, args.downsample_method)
    input_path = os.path.join(args.project_dir, "outputs")
    ply_path = os.path.join(input_path, list_pyramid[0])
    tile_aabb_list = scene.get_tiles(ply_path, args.max_splat_num)
    lod_tiles_save_path = os.path.join(output_path, "lod_tiles_info.txt")
    save_tile_info(lod_tiles_save_path, tile_aabb_list)
    
    # 对每个 tiles 生成 LOD 并保存成果
    tiles_json_info = []
    node_nums = {}
    print("Creating LOD ...")
    for i in range(len(tile_aabb_list)):
        tile_aabb = tile_aabb_list[i]
        tile_min_point, tile_max_point = tile_aabb[0], tile_aabb[1]
        json_info, node_num = create_single_LOD_pyramid(i, tile_min_point, tile_max_point, pyramid_to_level, input_path, output_path, 
                                                        ori_aabbs_info_dic, lod_level, R, args.max_splat_num, args.geometricError_method)  
        node_nums[f'tile{i}'] = node_num
        tiles_json_info.append(json_info)     
    
    node_num_path = os.path.join(output_path, "node_nums.txt")
    save_nodes_info(node_num_path, node_nums)
    
    # 生成 json 文件
    print("Creating json file ...")
    create_project_json(name, output_path, transform_matrix, scene_min_point, scene_max_point, tiles_json_info, 16.0)
    print("All done !!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("-p", "--project_dir", type=str, help="the project in", required=True, default="")
    parser.add_argument("--project_name", type=str, help="the project name", default="")
    parser.add_argument("-o", "--output_dir", type=str, help="output path for LOD", default="")
    parser.add_argument("-l","--list_pyramid", type=list_of_strings, help="the list of pyramid result used to create LOD", 
                        default=["scale_8", "scale_4", "scale_2", "scale_1"])

    parser.add_argument("-m", "--max_splat_num", type=int, help="max number of splats in a single chunk", default=61440)
    
    parser.add_argument("--downsample_list", type=list_of_strings, help="pyramid levels requiring downsampling", default=[])
    parser.add_argument("-d", "--downsample_ratio", type=float, default=0.5)
    parser.add_argument("--downsample_time", type=int, help="sampling times", default=1)
    parser.add_argument("--downsample_method", type=int, help="0", default=0)
    
    parser.add_argument("--devide_method", type=int, help="0", default=0)
    parser.add_argument("--geometricError_method", type=int, 
                        help="0, Calculate the radius based on x, y, z; 1, Calculate the radius based on x, y", 
                        default=0)
    args = parser.parse_args()

    def get_last_folder_name(path):
        """
        拆分路径以获得最后一个部分
        path: str, 路径名称
        """
        normalized_path = os.path.normpath(path)
        last_folder = os.path.basename(normalized_path)
        return last_folder

    # 获取项目名称
    if args.project_name != "":
        project_name = args.project_name
    else:    
        project_name = get_last_folder_name(args.project_dir)

    # LOD 成果输出路径
    if args.output_dir != "":
        output_path = args.output_dir
    else:
        output_path = os.path.join(args.project_dir, "outputs", "lod_result")
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok = True)
    
    create_LOD_with_pyramid(project_name, args.list_pyramid, args.downsample_list, output_path, args)