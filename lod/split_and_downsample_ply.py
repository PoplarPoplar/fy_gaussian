# 2024/9/7
#
# 生产 LOD 数据，涵盖对数据进行划分和采样

from argparse import ArgumentParser
from plyfile import PlyData
import os

from utils.aabb_utils import read_bounds_file
from utils.ply_utils import split_ply, downsample_ply

def split_and_downsample_ply_pyramid(input_path, tile_name, ratio, downsample_times, end_level, polygon, method, is_downsaple):
    """
    对点云数据按照 AABB 进行裁剪，并按比例进行多层采样
    input_path: str, 所有训练结果存储路径
    tile_name: str, 需要处理的 tile
    ratio: float, 采样比例
    downsample_times: int, 下采样次数
    end_level: int, 该金字塔对应的 LOD 结束的层级
    polygon: polygon, 裁剪的多边形区域 
    method: int, 采样方法， 0 为随机采样，目前只支持 0
    is_downsaple: bool, 是否需要对点云数据进行下采样， true 代表是
    """
    bounds_file_dir = os.path.join(input_path, tile_name)
    if not os.path.exists(os.path.join(bounds_file_dir, "center.txt")) or not os.path.exists(os.path.join(bounds_file_dir, "extent.txt")):
        assert False, f"There doesn't exist center.txt or extent.txt in {bounds_file_dir} !"
    center, extent = read_bounds_file(bounds_file_dir)
    
    try:
        ply_path = os.path.join(input_path, tile_name, "point_cloud", "iteration_final", "point_cloud.ply")
        plydata = PlyData.read(ply_path)
    except:
        ply_path = os.path.join(input_path, tile_name, "point_cloud", "iteration_30000", "point_cloud.ply")
        plydata = PlyData.read(ply_path)
    temp_path = os.path.join(input_path, tile_name, "lod_temp")
    os.makedirs(temp_path, exist_ok=True)
    
    # 将训练得到的点云根据 AABB 以及 polygon 进行裁切
    min_point, max_point = center - extent/2, center + extent/2
    plydata = split_ply(min_point, max_point, plydata, polygon)
    
    plydata.write(os.path.join(temp_path, f"point_cloud_{end_level}.ply"))
    
    if is_downsaple:
        # 根据 LOD 层数以及 downsample_ratio 对点云进行下采样
        for i in range(downsample_times):
            plydata = downsample_ply(plydata, ratio, method)
            plydata.write(os.path.join(temp_path, f"point_cloud_{end_level - i - 1}.ply"))
            
    return center, extent

def split_and_downsample_ply(input_path, bounds_path, tile_name, ratio, level, method, is_clip):
    """
    对点云数据按照 AABB 进行裁剪，并按比例进行多层采样
    input_path: str, 所有训练结果存储路径
    bounds_path: str, 所有 center.txt 以及 extent.exe 存储路径
    tile_name: str, 需要处理的 tile
    ratio: float, 采样比例
    level: int, 采样次数
    method: int, 采样方法， 0 为随机采样，目前只支持 0
    is_clip: bool, 是否需要对点云数据按照 AABB 进行裁剪， true 代表是
    """
    bounds_file_dir = os.path.join(bounds_path, tile_name)
    if not os.path.exists(os.path.join(bounds_file_dir, "center.txt")) or not os.path.exists(os.path.join(bounds_file_dir, "extent.txt")):
        assert False, f"There doesn't exist center.txt or extent.txt in {bounds_file_dir} !"
    center, extent = read_bounds_file(bounds_file_dir)
        
    ply_path = os.path.join(input_path, tile_name, "point_cloud", "iteration_30000", "point_cloud.ply")
    plydata = PlyData.read(ply_path)
    temp_path = os.path.join(input_path, tile_name, "lod_temp")
    os.makedirs(temp_path, exist_ok=True)
        
    # 将训练得到的点云根据 AABB 进行裁切
    if is_clip:
        min_point, max_point = center - extent/2, center + extent/2
        plydata = split_ply(min_point, max_point, plydata)
        
    plydata.write(os.path.join(temp_path, f"point_cloud_{level -1}.ply"))
    # 根据 LOD 层数以及 downsample_ratio 对点云进行下采样
    for i in range(level-1):
        plydata = downsample_ply(plydata, ratio, method)
        plydata.write(os.path.join(temp_path, f"point_cloud_{level -2 -i}.ply"))
            
    return center, extent

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    # TODO：有待根据最后的目录结构进行调整
    parser.add_argument("-i", "--input_dir", type=str, help="the path of ply located in", required=True, default="")
    parser.add_argument("-b", "--bounds_file_dir", type=str, help="find the path for extent and center file", required=True, default="")
    parser.add_argument("-t","--tile_name", type=str, help="the tile need to splited and downsampled", required=True, default="")

    parser.add_argument("-d", "--downsample_ratio", type=float, default=0.25)
    parser.add_argument("-l", "--level", type=int, help="the total number of LOD levels", default=3)

    parser.add_argument("--downsample_method", type=int, help="0", default=0)
    
    parser.add_argument("--is_clip", action='store_true', help="according to AABB clipping point cloud")
    parser.add_argument("--is_pyramid", action='store_true', help="whether to use image pyramid")
    args = parser.parse_args()
    
    center, extent = split_and_downsample_ply(args.input_dir, args.bounds_file_dir, args.tile_name, args.downsample_ratio,
                             args.level, args.downsample_method, args.is_clip, args.is_pyramid)