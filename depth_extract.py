import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
from torch.utils.data import DataLoader
import open3d as o3d            # open3d在python312版本下未更新   3.8-3.11均可以使用                         
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def direct_collate(x):
    return x

def write_dmb(file_path, data_map):
    try:
        with open(file_path, 'wb') as outimage:
            type = np.int32(1)
            h, w, c = data_map.shape
            nb = np.int32(c)

            outimage.write(type.tobytes())
            outimage.write(np.int32(h).tobytes())
            outimage.write(np.int32(w).tobytes())
            outimage.write(nb.tobytes())

            data = data_map.astype(np.float32).tobytes()
            outimage.write(data)
    except IOError:
        print(f"Error opening file {file_path}")

def load_camera_colmap(args):
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)


def extract_depth(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(0)
    scene = Scene(dataset, gaussians)
    output_path = os.path.join(dataset.model_path,"point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration= max(iteration,int(folder_name.split('_')[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(output_path,"iteration_"+str(iteration),"point_cloud.ply")

    gaussians.load_ply(output_path)
    print(f'Loaded gaussians from {output_path}')
    
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_cam_list = load_camera_colmap(dataset)

    # depth_list = []
    # color_list = []
    # normal_list = []
    alpha_thres = 0.5
    o3d_device = o3d.core.Device("CPU:0")
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 4, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

    merged_pcd = o3d.geometry.PointCloud()
    os.makedirs(os.path.join(dataset.model_path,"depth"), exist_ok=True)
    for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                # Rendering offscreen from that camera
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda() 
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy()
                # color_list.append(rendered_img)
                depth = render_pkg["median_depth"].clone()
                normal = render_pkg["normal"].clone()
                # if viewpoint_cam.gt_mask is not None:
                #     depth[(viewpoint_cam.gt_mask < 0.5)] = 0
                depth[render_pkg["mask"]<alpha_thres] = 0
                # depth_list.append(depth[0].cpu().numpy())
                # normal_list.append(normal.cpu().numpy())

                # 为了后续在open3d中进行处理
                color = rendered_img
                depth = depth[0].cpu().numpy()
                normal = normal.cpu().numpy()

                # depth = o3d.cuda.pybind.t.geometry.Image(depth)
                depth = o3d.t.geometry.Image(depth)
                depth = depth.to(o3d_device)
                W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
                fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
                fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
                intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
                # intrinsic = o3d.cuda.pybind.core.Tensor(intrinsic)
                # extrinsic = o3d.cuda.pybind.core.Tensor(viewpoint_cam.extrinsic.cpu().numpy().astype(np.float64))
                intrinsic = o3d.core.Tensor(intrinsic)

                R = viewpoint_cam.R
                T = viewpoint_cam.T
                T = T.reshape(-1, 1)
                # 将R和T组合成外参矩阵
                extrinsic_matrix = np.hstack((R, T))  # 水平堆叠R和T
                extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
                extrinsic = o3d.core.Tensor(extrinsic_matrix.astype(np.float64))

                color_tensor = np.transpose(color, (1, 2, 0))
                # 创建 Open3D 彩色图像
                color_image = o3d.geometry.Image(np.ascontiguousarray((color_tensor*255).astype(np.uint8)))
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, o3d.geometry.Image(depth), 1, 5000, False)
                #depth_image.set_data(depth.get_data())
                cam = o3d.camera.PinholeCameraIntrinsic(depth.columns, depth.rows, intrinsic[0][0].item(), intrinsic[1][1].item(), intrinsic[0][2].item(), intrinsic[1][2].item())
                #pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), cam, viewpoint_cam.extrinsic.cpu().numpy().astype(np.float64), 1, 5000, 4)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,cam, extrinsic_matrix.astype(np.float64))
                pcd = pcd.uniform_down_sample(10)

                #extrinsic = np.zeros([4,4],dtype=np.float32)
                #extrinsic = viewpoint_cam.extrinsic.cpu().numpy().astype(np.float64)
                #pcd.transform(np.linalg.inv(extrinsic))


                o3d.io.write_point_cloud(os.path.join(dataset.model_path,"depth",f"{viewpoint_cam.image_name}.ply"), pcd)
                #pcd.transform(viewpoint_cam.extrinsic.cpu().numpy().astype(np.float64))
                merged_pcd+=pcd
                depth_image_np = np.asarray(depth)
                write_dmb(os.path.join(dataset.model_path,"depth",f"{viewpoint_cam.image_name}-depth.dmb"), depth_image_np)

                normal = np.moveaxis(normal, 0, -1)  # 将通道移至最后
                #将法向量由摄像机坐标系转换到世界坐标系下
                rotation_matrix = extrinsic[:3, :3]

                # 将法向量从摄像机坐标系转换到世界坐标系
                normals_world = np.einsum('ij,hwj->hwi', rotation_matrix.numpy(), normal)

                # 归一化法向量
                normals_world /= np.linalg.norm(normals_world, axis=-1, keepdims=True)

                normal = normals_world

                write_dmb(os.path.join(dataset.model_path,"depth",f"{viewpoint_cam.image_name}-normal.dmb"), normal)
                
                # 归一化深度图像
                depth_normalized = (depth_image_np - np.min(depth_image_np)) / (np.max(depth_image_np) - np.min(depth_image_np))
                # 使用matplotlib的colormap进行伪彩色映射
                depth_colormap = plt.get_cmap('viridis')(depth_normalized)[:, :, :3]  # 取RGB通道
                # 将归一化后的彩色映射转换为8位的图像数据
                depth_colormap_8bit = (depth_colormap * 255).astype(np.uint8)
                # 创建一个Open3D的Image对象
                depth_colormap_8bit_contiguous = np.ascontiguousarray(depth_colormap_8bit)

                color_image = o3d.geometry.Image(np.squeeze(depth_colormap_8bit_contiguous,axis=2))
                # 创建一个新的geometry.Image对象
                depth_image = o3d.geometry.Image(depth_image_np)
                o3d.io.write_image(os.path.join(dataset.model_path,"depth",f"{viewpoint_cam.image_name}.png"), color_image)


                #norm = np.linalg.norm(normal, axis=2)
                #normal_normalized = normal / norm
                normal_normalized = normal
                # 步骤2: 彩色映射
                # 使用OpenCV的applyColorMap函数进行彩色映射
                #normal_normalized = np.moveaxis(normal_normalized, 0, -1)  # 将通道移至最后
                # 步骤1: 将数据范围从[-1, 1]调整到[0, 1]，以适应colormap的输入
                normal_normalized = (normal_normalized + 1) / 2

                # 步骤2: 使用matplotlib的colormap进行伪彩色映射
                # 创建一个与数据形状相匹配的空图像
                # color_mapped_normal = np.zeros((normal_normalized.shape[0], normal_normalized.shape[1], 3), dtype=np.uint8)

                # # 对每个通道应用colormap
                # for i in range(3):
                #     color_mapped_normal[:, :, i] = plt.cm.jet(normal_normalized[i])[:, :, 0] * 255

                # # 步骤3: 转换为8位图像数据
                # color_mapped_normal = color_mapped_normal.astype(np.uint8)
                # 步骤3: 转换为8位图像数据
                # 此时 color_mapped_normal 已经是8位图像数据

                normal_map_8 = normal_normalized*255
                normal_map_8 = normal_map_8.astype(np.uint8)
                # 步骤4: 创建o3d.geometry.Image
                normal_color = o3d.geometry.Image(np.ascontiguousarray(normal_map_8))
                o3d.io.write_image(os.path.join(dataset.model_path,"depth",f"{viewpoint_cam.image_name}-normal.png"), normal_color)

    o3d.io.write_point_cloud(os.path.join(dataset.model_path,f"merged.ply"), merged_pcd)

    print("done!")
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_depth(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    