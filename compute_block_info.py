#
# panhailong 2014/09/12
#
#from scene.pycolmap_wrapper import read_colmap_sfm
import sys
import argparse
from preprocess.read_write_model import *
import math
import json
from make_gs_chunks import get_block_bbox
import xml.etree.ElementTree as ET

def get_sfm_size(sfm_path, bbox = None):
    _, _, points3d = read_model(sfm_path, ext=".bin")
    if bbox is None:
        return len(points3d)
    else:
        corner_min = bbox[1][0].copy()
        corner_max = bbox[1][1].copy()
        _points3d =  np.array([p.xyz for key, p in points3d.items()])
        return (np.all(_points3d < corner_max, axis=-1) *  np.all(_points3d > corner_min, axis=-1)).sum()
def read_param(args):
    #读入Gaussian.json
    with open(f'{args.block_path}/BoundingBox.json', 'r') as file:
        data = json.load(file)
        args.layers = data["GaussianReconParam"]["layers"]
        args.scene_point_factor = data["GaussianReconParam"]["point_radio"]

def read_gsd(args):
    # 解析 XML 文件
    tree = ET.parse(f'{args.block_path}/at.xml')
    root = tree.getroot()

    # 获取 GSD 的值
    gsd = root.find('GSD').text

    # 输出 GSD 的值
    print(f'GSD: {gsd}')
    return float(gsd)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', required=True, default="")
    # 总的层数，如果没有指定了总的层数时，就根据top_point_size自动计算
    # 如果指定了层数则自动地计算每地层的点数
    # 1/8分辨率的那一层点数和sfm点数一样多，两层之间是4倍关系
    parser.add_argument('--layers', default=-1, type=int)
    #场景的点数的倍率，最精细一层的点数=scene_point_num_factor*sfm点数*64，[0.1~5.0]是比较合适的范围
    parser.add_argument('--scene_point_factor', default=1.0, type=float)
    # tile_length_factor
    parser.add_argument('--tile_length_factor', default=1.0, type=float)
   
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])
    
    #读入参数
    #read_param(args)

    #读入gsd   
    gsd = read_gsd(args)

    # 计算最细一层的tile的边长
    #5876是怎么来的？有专门的文档详细说明
    tile_edge = 5876*gsd*args.tile_length_factor

    # 读入包围盒
    bbox = get_block_bbox(args.block_path)
    # 估计包围盒内将来生出来的高斯点数

    # 包围盒的xy平面面积
    min_point = bbox[1][0]
    max_point = bbox[1][1]
    # 计算包围盒的面积
    bbox_area = (max_point[0] - min_point[0]) * (max_point[1] - min_point[1])
    #当每一个tile的点数为8_000_000
    one_tile_point_num = int(8_000_000*args.scene_point_factor)
    # 计算包围盒内高斯点数
    total_point_num = int(bbox_area / (tile_edge * tile_edge) * one_tile_point_num)

    finest_point_area_ratio = one_tile_point_num / (tile_edge * tile_edge)

    # 计算金字塔的层数
    # 如果没有指定层数，则根据top_point_size自动计算
    # 认为最顶层的理想点数是3_000_000
    top_point_size = 3_000_000
    if args.layers == -1:
        # 计算金字塔的层数, 限制在4到10层之间
        layer_num = min(10, max(4, math.ceil(math.log(total_point_num/top_point_size)/ math.log(4))))
    else:
        layer_num = args.layers

    # 计算每层金字塔对应的点数
    # 下一层的点数是上一层的1/4
    layer_infos = []
    for i in range(layer_num):
        # 计算每层金字塔对应的点数
        # 下一层的点数是上一层的1/4
        # 最细一层编号为0
        # 最精细的4层分别对应原始分辨率、1/2、1/4、1/8
        # 然后再后每层金字塔的分辨率是前一层的一半，但都会使用1/8的图像

        # 计算每层金字塔对应的点数
        total_point_num = int(total_point_num * (0.25 ** (layer_num-i-1)))
        # 计算每层金字塔对应的tile大小
        scale = int(2**(layer_num-i-1))
        tile_length = tile_edge * scale

        point_area_radio = finest_point_area_ratio * (0.25 ** (layer_num-i-1))
        if i == 0:
            point_num = total_point_num
        else:
            point_num = int(point_num / 4)
        resolution = 1
        if scale >=16:
            resolution = int(scale/8)
            scale = 8
        layer_infos.append({"layer": i,
                            "image_scale":scale,
                            "image_resolution": resolution,
                                "total_point_num": int(total_point_num * (0.25 ** (layer_num-i-1))),
                                "tile_length": tile_length,
                                #考虑到tile大小不一定均等（在边界的情况），所以这里记录点数面积比
                                "point_area_ratio": point_area_radio,
                                })
        
        # 每一层分块的

    block_info = dict()
    # block_info["tile_length"] = tile_edge
    block_info["layer_num"] = layer_num
    block_info["layer_infos"] = layer_infos
    with open(f"{args.block_path}/block_gs_info.json", "w") as f:
        json.dump(block_info, f, indent=2)
    print("Compute block info done.")