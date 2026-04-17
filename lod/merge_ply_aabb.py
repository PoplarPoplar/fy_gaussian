# 2024/11/05
# 支持非金字塔图像训练结果与金字塔训练结果，然后通过 -i 参数指定输入文件夹， 若为金字塔结果， 通过 -l 指定需要融合的金字塔层级
# 如需要指定输出目录，用 -o 参数，金字塔结果默认输出地址为 Project_path/outputs/clip_result
# 使用示例 python --merge_ply_aabb.ply -p D:/Users/Administrator/Desktop/jinzita -l layer_0,layer_1 --min_point -421.000940 -1396.002200 -300 --max-point 1222.20666 1243.008050 300

from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from multiprocessing import Pool
# from shapely.geometry import Point, Polygon
import numpy as np
import os
import json

from LOD_create import extract_chunk_names
from utils.ply2bin_utils import build_rotation

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

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
    e = [float(e[0]), float(e[1]), min(float(e[2]), 10000)]

    return np.array(c), np.array(e)

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    R: 旋转矩阵，形状为 (N, 3, 3)
    返回值是 (N, 4) 四元数数组，形状为 (N, 4)
    """
    N = R.shape[0]
    q = np.zeros((N, 4))
    
    # 计算旋转矩阵的迹
    trace = np.trace(R, axis1=1, axis2=2)
    
    # 处理 trace > 0 的情况
    mask = trace > 0
    s_positive = 0.5 / np.sqrt(trace[mask] + 1.0)
    q[mask, 0] = 0.25 / s_positive
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) * s_positive
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) * s_positive
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) * s_positive
    
    # 处理 trace <= 0 的情况
    mask_neg = ~mask
    r = R[mask_neg]

    r00, r01, r02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    r10, r11, r12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    r20, r21, r22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    
    neg_indices = np.where(mask_neg)[0]
    
    mask_r0 = (r00 > r11) & (r00 > r22)
    s_r0 = 2.0 * np.sqrt(1.0 + r00[mask_r0] - r11[mask_r0] - r22[mask_r0])
    indices_r0 = neg_indices[mask_r0]
    q[indices_r0, 0] = (r21[mask_r0] - r12[mask_r0]) / s_r0
    q[indices_r0, 1] = 0.25 * s_r0
    q[indices_r0, 2] = (r01[mask_r0] + r10[mask_r0]) / s_r0
    q[indices_r0, 3] = (r02[mask_r0] + r20[mask_r0]) / s_r0
    
    mask_r1 = (r11 > r22) & ~mask_r0
    s_r1 = 2.0 * np.sqrt(1.0 + r11[mask_r1] - r00[mask_r1] - r22[mask_r1])
    indices_r1 = neg_indices[mask_r1]
    q[indices_r1, 0] = (r02[mask_r1] - r20[mask_r1]) / s_r1
    q[indices_r1, 1] = (r01[mask_r1] + r10[mask_r1]) / s_r1
    q[indices_r1, 2] = 0.25 * s_r1
    q[indices_r1, 3] = (r12[mask_r1] + r21[mask_r1]) / s_r1
    
    mask_r2 = ~(mask_r0 | mask_r1)
    s_r2 = 2.0 * np.sqrt(1.0 + r22[mask_r2] - r00[mask_r2] - r11[mask_r2])
    indices_r2 = neg_indices[mask_r2]
    q[indices_r2, 0] = (r10[mask_r2] - r01[mask_r2]) / s_r2
    q[indices_r2, 1] = (r02[mask_r2] + r20[mask_r2]) / s_r2
    q[indices_r2, 2] = (r12[mask_r2] + r21[mask_r2]) / s_r2
    q[indices_r2, 3] = 0.25 * s_r2
    
    return q

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

def split_ply(min_point, max_point, ply, polygon = None):
    """
    根据包围盒裁切点云文件，并返回裁切结果。
    min_point: list, AABB 包围盒最小点的坐标
    extent: list, AABB 包围盒最大点的坐标
    ply: PlyData, 待裁切的点云
    polygon: polygon, 用于裁切点云的多边形
    """
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

class ori_AABB_info(NamedTuple):
    uid: str
    center: np.array
    axis: np.array

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

def load_ply_data(base_path, folder_name):
    try:
        plydata_path = os.path.join(base_path, folder_name, "point_cloud/iteration_30000", "point_cloud.ply")
        ply_data = PlyData.read(plydata_path)
    except:
        plydata_path = os.path.join(base_path, folder_name, "point_cloud/iteration_final", "point_cloud.ply")
        ply_data = PlyData.read(plydata_path)
    return ply_data

def merge_ply_data(vertices):
    ply_data = vertices[0]
    for ply_data_add in vertices[1:]:
        merged_vertices = np.concatenate((ply_data['vertex'].data, ply_data_add['vertex'].data))
        ply_data = PlyData([PlyElement.describe(merged_vertices, 'vertex')])
    return ply_data

def rot_ply(rotation, ply_data):
    '''
    对点云进行旋转
    '''
    plydata = ply_data['vertex']
    x, y, z = plydata['x'], plydata['y'], plydata['z']
    rot_0, rot_1, rot_2, rot_3 = plydata['rot_0'], plydata['rot_1'], plydata['rot_2'], plydata['rot_3']
    positions =np.einsum('ijk,ik->ij', rotation.transpose(0,2,1), np.vstack([x, y, z]).T)
    plydata['x'], plydata['y'], plydata['z'] = positions.T[0], positions.T[1], positions.T[2]
    R_matrices = build_rotation(np.vstack([rot_0, rot_1, rot_2, rot_3]).T)
    R_matrices = np.matmul(rotation.transpose(0,2,1), R_matrices)
    quaternions = rotation_matrix_to_quaternion(R_matrices)
    plydata['rot_0'], plydata['rot_1'], plydata['rot_2'], plydata['rot_3'] = quaternions.T[0], quaternions.T[1], quaternions.T[2], quaternions.T[3]
    plydata = np.array(plydata.data)
    return PlyData([PlyElement.describe(plydata, 'vertex')])

def merge_ply_interesting(project_path, input_path, list_3dgs, output_path, input_aabb_info, rot, polygon=None):
    min_point, max_point = input_aabb_info
   
    for level in list_3dgs:
        ori_aabbs_info = []
        read_file_path = os.path.join(input_path, level)
        chunk_info_path = os.path.join(project_path, "chunks", level, "task_list.txt")
        if os.path.exists(chunk_info_path):
            folders = extract_chunk_names(chunk_info_path)
        else:
            folders = [name for name in os.listdir(read_file_path) if os.path.isdir(os.path.join(read_file_path, name))]
        for folder in folders:
            bounds_file_dir = os.path.join(read_file_path, folder)
            if not os.path.exists(os.path.join(bounds_file_dir, "center.txt")) or not os.path.exists(os.path.join(bounds_file_dir, "extent.txt")):
                assert False, f"There doesn't exist center.txt or extent.txt in {bounds_file_dir} !"
            center, extent = read_bounds_file(bounds_file_dir)
            ori_aabb_info = ori_AABB_info(uid=folder, center=center, axis=extent)
            ori_aabbs_info.append(ori_aabb_info)
    
        target_aabb = find_intersecting_aabbs(input_aabb_info, ori_aabbs_info)
        vertices = []
        for aabb_info in target_aabb:
            ply_data = load_ply_data(read_file_path, aabb_info.uid)
            # TODO 重构这部分代码
            gobal_min_point, gobal_max_point = input_aabb_info
            #local_min_point, local_min_point是由center, extent确定的
            local_min_point = aabb_info.center - aabb_info.axis / 2
            local_max_point = aabb_info.center + aabb_info.axis / 2
            # gobal_min_point和local_max_point的交集
            min_point = np.maximum(gobal_min_point, local_min_point)
            max_point = np.minimum(gobal_max_point, local_max_point)
            ply_data = split_ply(min_point, max_point, ply_data, polygon)


            # 保存到lod_temp1文件夹
            temp_path = os.path.join(read_file_path, aabb_info.uid, "cut_ply")
            os.makedirs(temp_path, exist_ok = True)
            ply_data.write(os.path.join(temp_path, f"{aabb_info.uid}.ply"))
            vertices.append(ply_data)
        
        ply_data = merge_ply_data(vertices)
        # 旋转矩阵为单位矩阵则不去旋转
        if not np.array_equal(rot, np.array([[[1,0,0],[0,1,0],[0,0,1]]])):
            ply_data = rot_ply(rot, ply_data)
        os.makedirs(output_path, exist_ok = True)
        ply_data.write(os.path.join(output_path, f"{level}_point_cloud.ply"))

from LOD_create import create_gps_info, coord_transform
def generate_gps_info(project_dir, output_dir):
    BoundingBox_info_json_path = os.path.join(project_dir, "BoundingBox.json")
    if not os.path.exists(BoundingBox_info_json_path):
        assert False, f"There doesn't exist BoundingBox.json in {project_dir}!"
    BoundingBox_info = read_json_file(BoundingBox_info_json_path)
    
    # 获取地理信息并生成 transform 矩阵       
    normalizer = np.array([BoundingBox_info["Normalizer"]["origin"]["x"], 
                           BoundingBox_info["Normalizer"]["origin"]["y"],
                           BoundingBox_info["Normalizer"]["origin"]["z"]])
    # 获取坐标轴旋转四元数
    Quaternion = BoundingBox_info.get("Quaternion")
    rot = np.array([Quaternion['w'],Quaternion['x'],Quaternion['y'],Quaternion['z']]) if Quaternion else np.array([1, 0, 0, 0])
    R = build_rotation(rot[np.newaxis,:]) # 变为(1，3，3)
     
    CoordinateSystem_json_path = os.path.join(project_dir, "CoordinateSystem.json")
    if not os.path.exists(CoordinateSystem_json_path):
        source_epsg = 4547 # 默认采用 CGCS2000 / 3-degree Gauss-Kruger CM 114E (EPSG:4547) 坐标
    else:
        CoordinateSystem_info = read_json_file(CoordinateSystem_json_path)
        output_coord = CoordinateSystem_info["output"]["value"]
        source_epsg = output_coord.split()[0]
    lon, lat, altitude = coord_transform(normalizer, source_epsg)
    create_gps_info(output_dir, lon, lat, altitude)
    return R

if __name__ == "__main__":
    parser = ArgumentParser(description="根据输入的 AABB 裁剪点云")
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("-p", "--project_path", type=str, help="the project in", required=True, default="")
    parser.add_argument("-i", "--input_path", type=str, help="the project name", default="")
    parser.add_argument("-o", "--output_path", type=str, help="output path", default="")
    parser.add_argument("-l","--list_3dgs", type=list_of_strings, help="the list of 3DGS result used to clip", 
                        default=[])

    args = parser.parse_args()

    if args.input_path != "":
        input_path = args.input_path
    else:
        input_path = os.path.join(args.project_path, "outputs")
    
    # 裁剪结果存储路径
    if args.output_path != "":
        output_path = args.output_path
    else:
        output_path = os.path.join(args.project_path, "merge_output")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Output folder: {}".format(output_path))
    
    # 生成gps信息
    rot_matrix = generate_gps_info(args.project_path, output_path)
    #如果 args.list_3dgs为空，那么把output_path目录下的子目录都添加到list_3dgs中
    #要求子目录必须以layer_开头
    if len(args.list_3dgs) == 0:
        args.list_3dgs = [i for i in os.listdir(input_path) if i.startswith("layer_")]

    BoundingBox_info_json_path = os.path.join(args.project_path, "BoundingBox.json")
    if os.path.exists(BoundingBox_info_json_path):
        BoundingBox_info = read_json_file(BoundingBox_info_json_path)
        user_max_point = np.array([BoundingBox_info['CurrentBoundingBox']['max']['x'],
                               BoundingBox_info['CurrentBoundingBox']['max']['y'],
                               BoundingBox_info['CurrentBoundingBox']['max']['z']])
        user_min_point = np.array([BoundingBox_info['CurrentBoundingBox']['min']['x'],
                                BoundingBox_info['CurrentBoundingBox']['min']['y'],
                                BoundingBox_info['CurrentBoundingBox']['min']['z']])
        aabb_info = (user_min_point, user_max_point)
        GaussianReconParam = BoundingBox_info.get("GaussianReconParam")
        if GaussianReconParam and "polygon" in GaussianReconParam and "vertices" in GaussianReconParam["polygon"]:
            vertices = [(v["x"], v["y"]) for v in GaussianReconParam["polygon"]["vertices"]]
            polygon = np.array(vertices)
            # polygon = Polygon(vertices) if vertices else None
        else:
            polygon = np.array([])
        
        merge_ply_interesting(args.project_path, input_path, args.list_3dgs, output_path, aabb_info, rot_matrix, polygon)
    