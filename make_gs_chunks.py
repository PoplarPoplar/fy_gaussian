#
# panhailong 2014/09/12
#
#from scene.pycolmap_wrapper import read_colmap_sfm
import sys
import torch
from tqdm import tqdm
from os import makedirs
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, ArgumentParser
import random
import numpy as np
import argparse
import os
from preprocess.read_write_model import *
import json
from scene.dataset_readers import storePly, qvec2rotmat
import multiprocessing
from plyfile import  PlyData, PlyElement
import open3d as o3d

def direct_collate(x):
    return x


def read_coord(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()  # 读取文件内容
            numbers = data.split()  # 将数据分割成单独的数值
            numbers = [float(num) for num in numbers]  # 将字符串转换为浮点数
            return numbers
    except FileNotFoundError:
        print(f"Cannot find File {file_path}")
    except Exception as e:
        print(f"Exception: {e}")

def get_aabb(center, extent):
    """
    根据中心点和边长计算AABB的最小和最大点。

    参数:
    center : tuple
        包围盒的中心点，格式为 (x, y, z)。
    extent : tuple
        包围盒在x, y, z方向的边长，格式为 (width, height, depth)。

    返回:
    numpy.ndarray
        包含最小点和最大点的数组，格式为 [(min_x, min_y, min_z), (max_x, max_y, max_z)]。
    """
    # 计算最小点
    min_x = center[0] - extent[0] / 2
    min_y = center[1] - extent[1] / 2
    min_z = center[2] - extent[2] / 2

    # 计算最大点
    max_x = center[0] + extent[0] / 2
    max_y = center[1] + extent[1] / 2
    max_z = center[2] + extent[2] / 2

    chunk_min = np.array([min_x, min_y, min_z])
    chunk_max = np.array([max_x, max_y, max_z])

    return np.stack([chunk_min, chunk_max])

def get_i_j_from_name(chunk_name):
    # 从名称中提取数字部分
    parts = chunk_name.split('_')
    i_str, j_str = parts[1], parts[2]

    return int(i_str), int(j_str)

    
def make_chunk_list(input_chunk, chunk_size_x, chunk_size_y):
    """
    将输入的input_chunk在xy平面方向平均分成4个子chunks
    假如上层块的名称是chunk_001_002，每一个小块的命名规则如下：  
    分别为：
    chunk_001_003
    chunk_001_004
    chunk_002_003
    chunk_002_004
    也就是: 
    chunk_{2*(i-1)+1}_{2*(j-1)+1}
    chunk_{2*(i-1)+1}_{2*(j-1)+2}
    chunk_{2*(i-1)+2}_{2*(j-1)+1}
    chunk_{2*(i-1)+2}_{2*(j-1)+2}

    参数:
    input_chunk : tuple
        tuple的第0个元素为名称如："chunk_001_002"
        第二个元素为包含最小点和最大点的numpy.ndarray数组，格式为 [[min_x, min_y, min_z], [max_x, max_y, max_z]]。

    返回：
    list
    子chunks的列表，每一个元素都是一个tuple，格式和input_chunk一样
    """
    # global_bbox 是一个包含最小点和最大点的数组，形式如 np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    # chunk_size 是小包围盒的边长
    # 解包输入的chunk信息
    chunk_name, bounds = input_chunk
    # 从名称中提取i和j
    i, j = get_i_j_from_name(chunk_name)

    min_point = bounds[0]
    max_point = bounds[1]

    # 计算全局包围盒的中心点
    center = (min_point + max_point) / 2

    # 计算全局包围盒的尺寸
    dimensions = max_point - min_point

    # 计算在x和y方向上可以切分的块数
    num_chunks_x = int(np.ceil(dimensions[0] / chunk_size_x))
    num_chunks_y = int(np.ceil(dimensions[1] / chunk_size_y))

    # 初始化小包围盒的列表
    chunks = []
    # 遍历每个可能的块位置
    for sub_i in range(num_chunks_x):
        for sub_j in range(num_chunks_y):
            # 计算小包围盒
            chunk_min = min_point + np.array([sub_i*chunk_size_x,sub_j*chunk_size_y, 0])
            # 计算小包围盒的结束点，确保不超过大包围盒的边界
            chunk_max_x = min(chunk_min[0] + chunk_size_x, max_point[0])
            chunk_max_y = min(chunk_min[1] + chunk_size_y, max_point[1])
            chunk_max = np.array([chunk_max_x, chunk_max_y, max_point[2]])
            # 为小包围盒生成名称
            chunk_name = f"chunk_{2*(i-1)+(sub_i+1):03}_{2*(j-1)+(sub_j+1):03}"

            # 将小包围盒和名称添加到列表中
            chunks.append((chunk_name, np.stack([chunk_min, chunk_max])))

    return chunks
                

def get_block_bbox(block_path):
    "get the bbox of the whole block"
    bbox_filename = f"{block_path}/BoundingBox.json"
    with open(bbox_filename, 'r') as file:
        data = json.load(file)
        current_bounding_box = data['CurrentBoundingBox']
        # 提取Normalizer中的transform的rotation部分
        rotation_quaternion = np.array([data['Normalizer']['transform']['rotation']['w'],
                                    data['Normalizer']['transform']['rotation']['x'],
                                    data['Normalizer']['transform']['rotation']['y'],
                                    data['Normalizer']['transform']['rotation']['z']])

        
        # 将四元数转换为旋转矩阵
        rotation_matrix = qvec2rotmat(rotation_quaternion)

        # 将max和min点转换为numpy数组
        max_point = np.array([current_bounding_box['max']['x'], current_bounding_box['max']['y'], current_bounding_box['max']['z']])
        min_point = np.array([current_bounding_box['min']['x'], current_bounding_box['min']['y'], current_bounding_box['min']['z']])

        # 应用旋转到max和min点
        rotated_max = np.dot(rotation_matrix, max_point)
        rotated_min = np.dot(rotation_matrix, min_point)

        # 打印旋转后的Bounding Box
        print("Rotated Bounding Box Max:", rotated_max)
        print("Rotated Bounding Box Min:", rotated_min)

        #假想一个最顶层的块chunk_001_001
        global_bbox = ("chunk_001_001", np.stack([rotated_min, rotated_max]))
        return global_bbox

def get_tile_bbox(tile_path, chunk_name):
    center = read_coord(tile_path+'/center.txt')
    extent = read_coord(tile_path+'/extent.txt')
    #extent[2] = 100
    input_chunk = (chunk_name ,get_aabb(center, extent))
    return input_chunk

def read_sfm(sfm_path):
    cam_intrinsics, images_metas, points3d = read_model(sfm_path, ext=".bin")
    print("read colmap done")
    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32)
        for key in images_metas
    ])

    #valid_points_set = set(points3d.keys())  # 提取 points3d 的所有有效 key
    ## 遍历images_metas，把images_metas[key].point3D_ids中小于0的、不在points3d中的元素删除
    ## 由于images_metas不能直接修改，所以需要新建一个字典来存储修改后的数据
    # images_metas_new = {}
    # for key in images_metas:
    #     image_meta = images_metas[key]
    #     point3D_ids = image_meta.point3D_ids  # numpy array
    #     mask = (point3D_ids >= 0) & np.vectorize(valid_points_set.__contains__)(point3D_ids)  # 同时满足大于等于0和存在于points3d
    #     images_metas_new[key] = Image(
    #         id = key,
    #         qvec = image_meta.qvec,
    #         tvec = image_meta.tvec,
    #         camera_id = image_meta.camera_id,
    #         name = image_meta.name,
    #         xys = image_meta.xys[mask],
    #         point3D_ids = image_meta.point3D_ids[mask]  # 过滤后的数组
    #     )

    sfm = {}
    sfm["cam_intrinsics"] = cam_intrinsics
    sfm["cam_centers"] = cam_centers
    sfm["images_metas"] = images_metas
    sfm["points3d"] = points3d


   
    return sfm


def get_vaild_cam_by_sfm(sfm, chunk, min_pt_num = 50, candidate_cam = None):
    "根据图像观察到的sfm点是否在chunk内部，判断是否选择这个照片"

    corner_min = chunk[1][0].copy()
    corner_max = chunk[1][1].copy()
    corner_min[2] = -1e12
    corner_max[2] = 1e12
    select_image = {}


    points3d_dict = {key: point.xyz for key, point in sfm["points3d"].items()}
    all_points3d_keys = np.array(list(points3d_dict.keys()))
    all_points3d_values = np.array(list(points3d_dict.values()))

    # 排序 keys 和 values
    sorted_indices = np.argsort(all_points3d_keys)
    all_points3d_keys = all_points3d_keys[sorted_indices]
    all_points3d_values = all_points3d_values[sorted_indices]
    # 创建一个布尔掩码，用于选择在chunk内部的点
    all_points3d_select_mask = np.zeros(len(all_points3d_keys), dtype=bool)

    for image_idx, image_key in tqdm(enumerate(sfm["images_metas"]), total=len(sfm["images_metas"]), desc="Filter images."):
        if candidate_cam is None or candidate_cam[image_idx]:
            image_meta = sfm["images_metas"][image_key]
            # 当前图像的3D点
            #image_points3d =  np.array([sfm["points3d"][pt_key].xyz for pt_key in image_meta.point3D_ids])\
            # TODO: 排除掉image_meta.point3D_ids 为空的图像，目前是不会被选择的，后面可以根据它的 pose 来选择
            if len(image_meta.point3D_ids) == 0:
                print("image_meta.point3D_ids is empty")
                continue
            point_indices = np.searchsorted(all_points3d_keys, image_meta.point3D_ids)

            # 排除掉image_meta.point3D_ids中查找不到的3D点
            point_indices = np.clip(point_indices, 0, len(all_points3d_keys) - 1)
            vaild_point_mask = all_points3d_keys[point_indices] == image_meta.point3D_ids
            point_indices = point_indices[vaild_point_mask]

            image_points3d = all_points3d_values[point_indices]
            all_points3d_select_mask[point_indices] = True
            inbox_mask = ((image_points3d >= corner_min) & (image_points3d <= corner_max)).all(axis=-1)
            n_pts = inbox_mask.sum() if len(image_points3d) > 0 else 0
            # 一张照片有min_pt_num个以上的点的时候才加入到训练中
            #print(f"n_pts: {n_pts}")
            if n_pts > min_pt_num:
                # 只有出现过的point3D_ids，才需要保存，对应的xys也要保存
                select_image[image_key] = Image(
                    id = image_key,
                    qvec = image_meta.qvec,
                    tvec = image_meta.tvec,
                    camera_id = image_meta.camera_id,
                    name = image_meta.name,
                    xys = image_meta.xys,
                    point3D_ids = image_meta.point3D_ids
                )

    # 筛选Point3D
    # 将 select_image 的键转换为已排序的 NumPy 数组
    select_image_keys = np.array(list(select_image.keys()))
    select_image_keys.sort()  # 确保排序

    # 创建 select_point3d
    select_point3d = {}

    if len(select_image_keys) > 0:
        for point3d_key in tqdm(all_points3d_keys[all_points3d_select_mask], desc="Processing Point3D"):
            point3d_value = sfm["points3d"][point3d_key]

            # 使用 np.searchsorted 查找 image_ids 是否在 select_image_keys 中
            indices = np.searchsorted(select_image_keys, point3d_value.image_ids)
            indices = np.clip(indices, 0, len(select_image_keys) - 1)  # 限制索引范围
            mask = select_image_keys[indices] == point3d_value.image_ids

            # 如果 mask 中存在 True，则说明 point3d_value 在 select_image 中
            if mask.any():
                select_point3d[point3d_key] = Point3D(
                    id=point3d_value.id,
                    xyz=point3d_value.xyz,
                    rgb=point3d_value.rgb,
                    error=point3d_value.error,
                    image_ids=point3d_value.image_ids[mask],
                    point2D_idxs=point3d_value.point2D_idxs[mask]
                )
    return select_image, select_point3d


def save_chunk_data(sfm, gs, chunk, args, output_dir):
    corner_min = chunk[1][0]
    corner_max = chunk[1][1]
    # corner_min[2] = -1e12
    # corner_max[2] = 1e12
    
    box_center = (corner_max + corner_min) / 2
    extent = (corner_max - corner_min) / 2

    # 扩展包围盒，以包含所有可能的相机pos
    acceptable_radius = 20
    extended_corner_min = box_center - acceptable_radius * extent
    extended_corner_max = box_center + acceptable_radius * extent

    # 一张照片，如果距离摄像机过远（20倍包围盒）的话，也不需要
    valid_cam = np.all(sfm["cam_centers"] < extended_corner_max, axis=-1) * np.all(sfm["cam_centers"] > extended_corner_min, axis=-1)

    #使用sfm点来确定这个相机是否被选中
    select_image, select_point = get_vaild_cam_by_sfm(sfm, chunk, args.min_pt_num, valid_cam)
    
    print(f"{chunk[0]}: {len(select_image)} valid cameras {len(select_point)} valid points")
    if len(select_image) < args.min_n_cams or len(select_point) < len(select_image)*5:
        return False

    out_colmap = os.path.join(output_dir, "sparse", "0")
    os.makedirs(out_colmap, exist_ok=True)


    print("saving points3D.ply")
    ply_xyz=[]
    ply_rgb=[]
    for pt_key, pt_value in select_point.items():
        ply_xyz.append(pt_value.xyz)
        ply_rgb.append(pt_value.rgb)
    
    ply_xyz = np.vstack(ply_xyz)
    ply_rgb = np.vstack(ply_rgb)

    if gs is not None:
        # 从高斯点云，提取点云，和sfm合并在一起保存
        # aabb内部的gs点, 1:1采样
        # 用于框选高斯的aabb 外扩0.1
        gs_corner_min = box_center - extent * 1.1
        gs_corner_max = box_center + extent * 1.1
        gs_filter = np.all(gs["xyz"] < gs_corner_max, axis=-1) *  np.all(gs["xyz"] > gs_corner_min, axis=-1)
        # aabb外部的gs点, 1:10采样
        #gs_filter = np.logical_or(gs_filter, np.random.rand(len(gs["xyz"])) < 0.1)

        # 合并到ply_xyz中
        ply_xyz = np.vstack([ply_xyz, gs["xyz"][gs_filter]])
        ply_rgb = np.vstack([ply_rgb, gs["rgb"][gs_filter]])
    # 保存点云
    storePly(f"{out_colmap}/points3D.ply", ply_xyz, ply_rgb)
    print("saving points3D.ply done")

    print("saving chunk sfm")
    write_model(sfm["cam_intrinsics"], select_image, select_point, out_colmap, f".{args.model_type}")


    with open(os.path.join(output_dir, "center.txt"), 'w') as f:
        f.write(' '.join(map(str, (corner_min + corner_max) / 2)))
    with open(os.path.join(output_dir, "extent.txt"), 'w') as f:
        f.write(' '.join(map(str, corner_max - corner_min)))
    return True
   

def save_valid_chunks(file_path, vaild_chunks):
     with open(file_path, "w") as file:
        file.write(str(len(vaild_chunks)) + "\n")
        # 遍历数组，写入每个元素
        for item in vaild_chunks:
            file.write(item + "\n")

def read_valid_chunks(file_path):
    with open(file_path, "r") as file:
        # 读取第一行，它包含了有效块的数量
        num_valid_chunks = int(file.readline().strip())
        # 初始化一个列表来存储有效块
        valid_chunks = []
        # 遍历文件的剩余部分，读取每个有效块
        for line in file:
            # 去除每行末尾的换行符，并添加到列表中
            valid_chunks.append(line.strip())
        # 返回有效块的数量和列表
        return num_valid_chunks, valid_chunks

def get_layer_by_scale(args):
    #读入block_gs_info.json
    with open(f'{args.block_path}/block_gs_info.json', 'r') as file:
        data = json.load(file)

        for item in data:
            if 2**(3-item['pyramid_layer']) == args.scale:
                return item['layer']        
    #从读入block_gs_info.json匹配当前的scale得到layer
    return -1


def read_tile_info(args):
    #读入tile_length.json
    with open(f'{args.block_path}/block_gs_info.json', 'r') as file:
        data = json.load(file)
        if args.layer >= data['layer_num']:
            print("layer is out of range")
            #TODO 将来需要把所有的异常情况都处理一下
            sys.exit(0)
        return data['layer_infos'][args.layer]['tile_length']

def run(args):
    chunk_in_dir = f"{args.block_path}/chunks/layer_{args.layer-1}"
    chunk_out_dir = f"{args.block_path}/chunks/layer_{args.layer}"
    os.makedirs(chunk_out_dir, exist_ok=True)

    args.chunk_size= read_tile_info(args)

    """
    流程：
    1. 读入包围盒（如果是最高层layer_0则读测区的bounding box，其他情况则读各自的center和extend）
    2. 对包围分成若干块（如果是layer_0则按用户指定大小分，其他情况则均分成2x2的块）
    3. 切出每一个块中的小sfm并保存
    4. 如果不是layer_0，则还要切出每一个块中的高斯点并保存
    """
    if args.layer == 0:
        sfm_path = f"{args.block_path}"
    else:
        sfm_path = f"{chunk_in_dir}/{args.input_chunk_name}"
    #1. 读入包围盒（如果是最高层则读测区的bounding box，其他情况则读各自的center和extend）
    if args.layer == 0:
        input_chunk = get_block_bbox(sfm_path)
    else:
        input_chunk = get_tile_bbox(sfm_path, args.input_chunk_name)

    #2. 对包围分成若干块
    chunk_size_x = args.chunk_size
    chunk_size_y = args.chunk_size
   

    chunk_list = make_chunk_list(input_chunk, chunk_size_x, chunk_size_y)
    print(f"Get {len(chunk_list)} chunks")

    for chunk in chunk_list:
        temp = chunk[1]
        # temp[0][2] = -200
        # temp[1][2] = 200
        storePly(f"{chunk_out_dir}/box_{chunk[0]}.ply", temp, np.zeros((temp.shape[0], 3), dtype=np.uint8))

   
    # 读入空三    
    print("read_sfm start")
    sfm = read_sfm(f"{sfm_path}/sparse/0")
    print(f"read_sfm done, images = {len(sfm['images_metas'])}")
    gs = None
    if args.layer != 0:
        gs = {}
        # 读入上一个layer的高斯
        args.init_gs_path = f"{args.block_path}/outputs/layer_{args.layer-1}/{args.input_chunk_name}/point_cloud/iteration_final/point_cloud.ply"
        plydata = PlyData.read(args.init_gs_path)
        gs["xyz"] = np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])) 
        gs["rgb"] = np.column_stack((plydata['vertex']['f_dc_0'], plydata['vertex']['f_dc_1'], plydata['vertex']['f_dc_2']))
        gs["rgb"] = (0.5+gs["rgb"]*0.28)*255
        gs["rgb"] = np.clip(gs["rgb"], 0, 255).astype(np.uint8)
        #storePly(f"D:/1.ply", gs["xyz"], gs["rgb"])
        # 创建 Open3D 点云对象
        print("create pcd start")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gs["xyz"])
        pcd.colors = o3d.utility.Vector3dVector(gs["rgb"] / 255.0)  # Open3D 颜色需要归一化到[0,1]

        # Step 1: 体素化下采样
        # 网格的大小，值越小保留的点越多
        print(f"Applying Voxel Grid Downsampling with voxel size {args.voxel_size}")
        down_pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        # 更新 gs 数据
        gs["xyz"] = np.asarray(down_pcd.points)
        gs["rgb"] = (np.asarray(down_pcd.colors) * 255).astype(np.uint8)  # 反归一化回到[0,255]
        #storePly(f"D:/2.ply", gs["xyz"], gs["rgb"])

        # # Step 1: 应用 Statistical Outlier Removal 滤波
        # print(f"Applying Statistical Outlier Removal")
        # _, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        # sor_pcd = down_pcd.select_by_index(ind)
        # #保存ply
        # gs["xyz"] = np.asarray(sor_pcd.points)
        # gs["rgb"] = (np.asarray(sor_pcd.colors) * 255).astype(np.uint8)  # 反归一化回到[0,255]
        # storePly(f"D:/3.ply", gs["xyz"], gs["rgb"])



    # 3. 获得每一个块中的小sfm
    #将sfm划分成到每一个训练块中
    invaild_chunks = []
    vaild_chunks = []
    for chunk in chunk_list:
        if save_chunk_data(sfm, gs, chunk, args, f"{chunk_out_dir}/{chunk[0]}"):
            vaild_chunks.append(chunk[0])
        else:
            invaild_chunks.append(chunk[0])

    # 使用多进程池并发处理 tiles
    # pool = multiprocessing.Pool(processes=min(args.multi_process, multiprocessing.cpu_count()))
    # results = {chunk[0]: pool.apply_async(save_chunk_data, (sfm, gs, chunk, args, f"{chunk_out_dir}/{chunk[0]}")) 
    #            for chunk in chunk_list}
    # pool.close()
    # pool.join()
    # # 获取结果
    # for chunk_name, result in results.items():
    #     if result.get():
    #         vaild_chunks.append(chunk_name)
    #     else:
    #         invaild_chunks.append(chunk_name)

    chunk_list_path = ""
    if args.layer == 0:
        chunk_list_path = f"{chunk_out_dir}/vaild_chunks.txt"
    else:
        chunk_list_path = f"{chunk_out_dir}/vaild_chunks-{args.input_chunk_name}.txt"
    save_valid_chunks(chunk_list_path, vaild_chunks)
    
    with open(f"{chunk_out_dir}/invaild_chunks-{args.input_chunk_name}.json", "w") as f:
        json.dump(invaild_chunks, f, indent=2)
if __name__ == '__main__':
    # 打包时需要加入下面这句，否则 pyinstaller 打包后的 exe 遇到多进程语句就会重新初始化无法正确读入参数
    # 使用py脚本运行则需要注释这句
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method("spawn", force=True)


    random.seed(0)
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--block_path', required=True, default="")
    parser.add_argument('--min_n_cams', default=30, type=int) # 
    parser.add_argument('--min_pt_num', default=20, type=int) # 
    parser.add_argument('--model_type', default="bin")
    parser.add_argument("--input_chunk_name", default="",type=str)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--voxel_size', default=0.5, type=float)
    parser.add_argument("--multi_process", type=int, help="max number of processes", default=8)

    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])
    
    #检查参数
    if args.layer  < 0 or args.layer > 10:
        print("Invaild --layer")
    if args.layer != 0 and len(args.input_chunk_name) == 0:
        print("Please input --input_chunk_name")
   
    run(args)

    print("Make gs chunks done.")