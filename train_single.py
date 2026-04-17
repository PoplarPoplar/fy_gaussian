# 功能：执行单个高斯分块的最小训练流程，支持在缺少 fused_ssim 扩展时自动回退到 PyTorch SSIM。
#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time, shutil
import numpy as np
from adapt_split_speed_contral import AdaptiveSplitting
from torch.utils.tensorboard import SummaryWriter
import datetime
import lpips

from PIL import Image
from utils.graphics_utils import point_double_to_normal, depth_double_to_normal

try:
    from fused_ssim import fused_ssim
except ImportError:
    fused_ssim = None

def visualize_tensor(img_tensor, file_path):
    """
    可视化 PyTorch tensor 图像

    参数:
    img_tensor (torch.Tensor): 大小为 (1, H, W) 的 PyTorch tensor

    返回:
    None
    """
    # 检查 tensor 的形状
    if img_tensor.ndim != 3 or img_tensor.shape[0] != 1:
        raise ValueError("输入 tensor 的形状必须为 (1, H, W)")

    # 将 tensor 转换为 numpy 数组
    img_np = img_tensor.detach().squeeze().cpu().numpy()
    img_pil = Image.fromarray(img_np)
    img_pil.save(file_path, format='TIFF')

def direct_collate(x):
    return x



def get_stage_params(stage_iterations, iteration):
    # 把训练分成2个阶段，使用2种策略
    # 第一阶段：4N选N, 目标点数为GPU能容纳最大点数 max_gs_num
    #          结束后，在开始第二阶段之前，按view_num > 6 删点策略, 然后再使用随机删点把高斯数降低到 0.5*max_gs_num_in_box
    #          第阶段结束后，包围盒包的点就不要分裂了
    # 第二阶段：从 16选N, 目标点数为 max_gs_num_in_box

    #          
 
    strategy = 0
    if iteration < stage_iterations[0]:
        strategy = 0
    elif iteration < stage_iterations[1]:
        strategy = 1
    else:
        strategy = 2
   
    return strategy



def training(dataset, opt, pipe, output_milestone = False, writer = None):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.set_writer(writer)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Ll1normal_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None  
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 4, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

    iteration = first_iter

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    
    # 自动计算分裂速度
    adaptive_splitting = AdaptiveSplitting()
    adaptive_splitting.set_params(final_num_points = opt.max_gs_num, exclude_num_points = gaussians.skybox_points,  
                                  iteration_end = opt.stage_iterations[0], 
                                  densification_interval = opt.densification_interval)

    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                background = torch.rand((3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                if not args.disable_viewer:
                    if network_gui.conn == None:
                        network_gui.try_connect()
                    while network_gui.conn != None:
                        try:
                            net_image_bytes = None
                            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                            if custom_cam != None:
                                if keep_alive:
                                    net_image = render(custom_cam, gaussians, pipe, background, require_coord=False, require_depth=False)["render"]
                                else:
                                    net_image = render(custom_cam, gaussians, pipe, background, require_coord=False, require_depth=False)["depth"].repeat(3, 1, 1)
                                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                            network_gui.send(net_image_bytes, dataset.source_path)
                            #if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                                #break
                        except Exception as e:
                            network_gui.conn = None

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                
                # Render
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, require_coord=False, require_depth=False)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                if len(visibility_filter) == 0:
                    print("No points visible, skipping iteration")
                    continue
                writer.add_scalar("Num/visibility_filter", visibility_filter.shape[0], iteration)
                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask
                
                # #可以（大幅）改善地面半透明的问题
                # if iteration > opt.densify_until_iter:
                #     opt.lambda_dssim = 1.0
                Ll1 = l1_loss(image, gt_image)
                if pipe.fused_ssim and fused_ssim is not None:
                    Lssim = (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
                else:
                    Lssim = (1.0 - ssim(image, gt_image))
                photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim 
                loss = photo_loss.clone()

                # 不再输入深度图  rade_gs cuda不再返回invDepth
                Ll1depth_pure = 0.0
                Ll1depth = 0
                # if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                #     mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                #     depth_mask = viewpoint_cam.depth_mask.cuda()

                #     Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                #     Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                #     loss += Ll1depth
                #     Ll1depth = Ll1depth.item()
                # else:
                #     Ll1depth = 0

                Ll1normal = 0
                if opt.normal_from_iter > -1 and iteration > opt.normal_from_iter and viewpoint_cam.normal_map is not None:
                    gt_normal = viewpoint_cam.normal_map.cuda()
                    normal = render_pkg["normal"]
                    gt_normal = gt_normal * viewpoint_cam.normal_mask.cuda()
                    normal = normal * viewpoint_cam.normal_mask.cuda()
                    Ll1normal = l1_loss(- gt_normal, normal)  # stable normal
                    # Ll1normal = l1_loss(gt_normal, normal)   # metric3d normal
                    loss += Ll1normal

                # Rade Gs
                reg_kick_on = opt.regularization_from_iter > -1 and iteration >= opt.regularization_from_iter  
                require_depth = True
                if reg_kick_on:
                    lambda_depth_normal = opt.lambda_depth_normal
                    if require_depth:
                        rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                        rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                        rendered_normal: torch.Tensor = render_pkg["normal"]
                        depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
                    else:
                        rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                        rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                        rendered_normal: torch.Tensor = render_pkg["normal"]
                        depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord, rendered_median_coord)
                    depth_ratio = 0.6
                    normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1)) * viewpoint_cam.normal_mask.cuda()
                    depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
                    loss += depth_normal_loss * lambda_depth_normal
                else:
                    lambda_depth_normal = 0
                    depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
                
                loss.backward()
                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                    ema_Ll1normal_for_log = 0.4 * Ll1normal + 0.6 * ema_Ll1normal_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", 
                                                  "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", 
                                                  "Normal Loss": f"{ema_Ll1normal_for_log:.{7}f}",
                                                  "Rade Loss": f"{depth_normal_loss.item():.{7}f}",
                                                  "Size": f"{gaussians._xyz.size(0)}"})
                        progress_bar.update(10)
                        # 将loss、点数记录到tensorboard中
                        if writer:
                            writer.add_scalar("Loss/Photo Loss", photo_loss.item(), iteration)
                            writer.add_scalar("Loss/Depth Loss", Ll1depth, iteration)
                            writer.add_scalar("Loss/Total Loss", loss.item(), iteration)
                            writer.add_scalar("Num/Point Number", gaussians._xyz.size(0), iteration)
                    if len(dataset.progress_path) and iteration % 1000 == 0:
                        progress_ratio = int(iteration / opt.iterations * 100)
                        with open(os.path.join(dataset.progress_path, 'progress'), 'w') as f:
                            f.write(f"{progress_ratio}")

                    # Log and save
                    # if (iteration in saving_iterations):
                    #     print("\n[ITER {}] Saving Gaussians".format(iteration))
                    #     scene.save(iteration)
                    #     print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    # 每1000次迭代保存一次模型
                    # if iteration % 1000 == 0:
                    #     print("\n[ITER {}] Saving Gaussians".format(iteration))
                    #     scene.save(iteration)
                    #     print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))
                  
                   
                    if iteration == opt.iterations:
                        scene.save("final")
                        progress_bar.close()
                        return

                
                    # Densification
                    if iteration < opt.densify_until_iter and gaussians._xyz.size(0) < opt.max_gs_num:
                        # Keep track of max radii in image-space for pruning   
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                     
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            grow_speed = adaptive_splitting.update(gaussians._xyz.size(0), iteration)
                            stage = get_stage_params(opt.stage_iterations, iteration)
                            gaussians.densify_and_prune(0.005, 0.005, grow_speed, stage)

                        if iteration in opt.opacity_reset_iters or (dataset.white_background and iteration == opt.densify_from_iter):
                            print("-----------------RESET OPACITY!-------------")
                            gaussians.reset_opacity()
                    
                   
                    # Optimizer step
                    if iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                        if gaussians._xyz.grad != None and gaussians.skybox_locked:
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                        elif gaussians._xyz.grad != None:
                            # 为了在训练单块时保留天空球skybox_locked为False,并且与skybox_locked为True的情况不产生冲突
                            # 以下注释可以根据需求是否改变天空球颜色
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            # gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            # gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            # gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                       
                        if gaussians._opacity.grad != None and gaussians._xyz.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if(relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            else:
                                gaussians.optimizer.step(relevant)
                                print("No grads!")
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    # 球变成椭球
                    if (gaussians.get_xyz.shape[0] - gaussians.skybox_points > 4 * gaussians.init_point_num or iteration == opt.sphere_to_ellipsoid_iter)  and gaussians.use_sphere:
                        #在变成椭球之前，先保存一下当前模型
                        if output_milestone:
                            scene.save("step_1")
                        gaussians.sphere_to_ellipsoid()
                    
                    #在第一、二阶段结束之后，使用删点策略
                    view_num = 0
                    if iteration == opt.stage_iterations[0]:
                        #gaussians.sphere_to_ellipsoid()
                        #scene.save("step_1")
                        view_num = 1
                    elif iteration == opt.stage_iterations[1]:
                        #scene.save("step_2")
                        view_num = 1
                    
                    
                    # 加入删点策略
                    if view_num > 0:
                        print(f"\nBegin to filter out points in iteration {iteration}")
                        max_weight_pids_sum = None
                        for delete_batch in training_generator:
                            for cam in delete_batch:
                                cam.world_view_transform = cam.world_view_transform.cuda()
                                cam.projection_matrix = cam.projection_matrix.cuda()
                                cam.full_proj_transform = cam.full_proj_transform.cuda()
                                cam.camera_center = cam.camera_center.cuda()
                                max_weights_pid = render(cam, gaussians, pipe, background, require_coord=False, require_depth=False, require_max_weight_indices=True)["maxweight_indices"].flatten()
                                counts = torch.bincount(max_weights_pid, minlength=gaussians.get_xyz.shape[0]).clamp_max(1)    # 变成照片数只有0, 1
                                if max_weight_pids_sum is None:
                                    max_weight_pids_sum = counts
                                else:
                                    max_weight_pids_sum += counts
                        
                        points_mask = max_weight_pids_sum < view_num
                        # 不删除脚手架或者天空球上的点
                        if gaussians.scaffold_points:
                            points_mask[:gaussians.scaffold_points] = False
                        points_mask[:gaussians.skybox_points] = False
                        print(f"\nBefore filter out point with visibility, the shape is {gaussians.get_xyz.shape[0]}")
                        if iteration == opt.stage_iterations[0]:
                            gaussians.prune_points(points_mask)
                            gaussians.simplify_points(int(0.5*opt.max_gs_num_in_box))
                        elif iteration == opt.stage_iterations[1]:
                            gaussians.prune_points(points_mask)
                            gaussians.simplify_points(int(0.5*opt.max_gs_num_in_box))
                       #                    # 
                        print(f"After filter out point with visibility, the shape is {gaussians.get_xyz.shape[0]}")
            # 
               
                        #第一、二阶段结束之后，调整分裂目标
                        if iteration == opt.stage_iterations[0]:
                            adaptive_splitting.set_params(final_num_points = opt.max_gs_num , exclude_num_points = gaussians.skybox_points + gaussians.get_outside_gs_num(),  
                                    iteration_end = opt.stage_iterations[1], 
                                    densification_interval = opt.densification_interval)
                        elif iteration == opt.stage_iterations[1]:
                            adaptive_splitting.set_params(final_num_points = opt.max_gs_num_in_box, exclude_num_points = gaussians.skybox_points + gaussians.get_outside_gs_num(),
                                    iteration_end = opt.stage_iterations[2], 
                                    densification_interval = opt.densification_interval)


                   
                    #删除超大的高斯,经过测试这个策略搭配球形高斯才有交，所以只在球形高斯阶段启用
                    # if iteration in opt.remove_big_gs_iter:
                    #     gaussians.remove_big_scale()
                    # if iteration in opt.opacity_remove_iter:
                    #     prune_mask = (gaussians.get_opacity < 0.1).squeeze()
                    #     gaussians.prune_points(prune_mask)
                   
                    if not args.skip_scale_big_gauss:
                        with torch.no_grad():
                            vals, _ = gaussians.get_scaling.max(dim=1)
                            violators = vals > scene.cameras_extent * 0.02
                            if gaussians.scaffold_points is not None:
                                violators[:gaussians.scaffold_points] = False
                            
                            # 同样为了在训练单块时保留天空球(不加载scaffold file时保留天空球的scale)
                            violators[:gaussians.skybox_points] = False
                            gaussians._scaling[violators] = gaussians.scaling_inverse_activation(gaussians.get_scaling[violators] * 0.8)
                
                    current_reserved  = torch.cuda.memory_reserved(device)
                    writer.add_scalar('memory_reserved', current_reserved, iteration)
                    #每1000步，释放一次缓存，防止显存峰值占用太多
                    if iteration % 1000 == 0:
                        torch.cuda.empty_cache()
                    iteration += 1


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    if len(args.progress_path):
        os.makedirs(args.progress_path, exist_ok=True)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
def read_colmap_file(args):
    images_folder = args.images
    depths_folder = args.depths
    try:
        cameras_extrinsic_file = os.path.join(args.source_path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(args.source_path, "sparse/0", "images.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    
    # depth_params_file = os.path.join(args.source_path, "sparse/0", "depth_params.json")
    # image_path_list = [cam_extrinsics[id].name for id  in cam_extrinsics]
    
    images_name_list = []    # 注意要与之前的images_folder相结合才是完整的路径
    depths_name_list = []

    for _, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        n_remove = len(extr.name.split('.')[-1]) + 1

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.jpg")
            image_name = f"{extr.name[:-n_remove]}.jpg"
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.png")
            image_name = f"{extr.name[:-n_remove]}.png"

        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        depth_name = f"{extr.name[:-n_remove]}.png"

        images_name_list.append(image_name)
        depths_name_list.append(depth_name)

    return images_name_list, depths_name_list

def get_train_image_num(args):
    images_name_list, _ = read_colmap_file(args)
    return len(images_name_list)

def copy_images_depths(args, images_name_list, depths_name_list):
    os.makedirs(os.path.join(args.cache_dir, "images"), exist_ok=True)
    if len(args.depths) > 0:
        os.makedirs(os.path.join(args.cache_dir, "depths"), exist_ok=True)
    for idx in range(len(images_name_list)):
        image_name = images_name_list[idx]
        depth_name = depths_name_list[idx]

        # 复制图片
        des_path = os.path.join(args.cache_dir, "images", image_name)
        src_path = os.path.join(args.images, image_name) 
        shutil.copy(src_path, des_path)

        if len(args.depths) > 0:
            # 复制深度图
            des_path = os.path.join(args.cache_dir, "depths", depth_name)
            src_path = os.path.join(args.depths, depth_name) 

            try:
                # Attempt to copy the file
                shutil.copy(src_path, des_path)
            except Exception as e:
                # Catch other possible exceptions
                print(f"Warning: An error occurred while copying the file. Error message: {e}")

        
        if (idx+1)%50 == 0:
            print(f"Already process {idx+1}")

def copy_normals(args, images_name_list):
    if len(args.normals) == 0:
        return 
    os.makedirs(os.path.join(args.cache_dir, "normals"), exist_ok=True)
    for idx in range(len(images_name_list)):
        image_name = images_name_list[idx]

        # 复制法向图
        des_path = os.path.join(args.cache_dir, "normals", image_name)
        src_path = os.path.join(args.normals, image_name) 
        shutil.copy(src_path, des_path)
        
        if (idx+1)%50 == 0:
            print(f"Already process normals {idx+1}")
    
def cache_train_data(args):
    if not args.do_cache:
        return
    args.cache_dir = f"{args.cache_dir}\\{args.gpu_id}\\{os.path.basename(os.path.normpath(args.source_path))}" 
    tt = time.time()
    # 0.可选是否复制空三文件，并且更改args.source 默认为True
    if args.copy_sfm:
        colmap_dir = os.path.join(args.cache_dir, "sfm")
        if os.path.exists(colmap_dir):
            shutil.rmtree(colmap_dir)
        shutil.copytree(args.source_path, colmap_dir)   # 这要求目标目录不存在，因为copytree()不会覆盖已存在的目录
        args.source_path = colmap_dir
    
    # 1.读取空三文件对应的路径
    images_name_list, depths_name_list = read_colmap_file(args)

    # 2.复制图片以及深度图到指定位置
    copy_images_depths(args, images_name_list, depths_name_list)

    args.images = f"{args.cache_dir}/images"
    args.depths = f"{args.cache_dir}/depths" if len(args.depths) > 0 else ""
    args.source_path = f"{args.cache_dir}/sfm"

    print(f"\nFinally, {len(images_name_list)} images take {time.time()-tt} seconds")

    # 3. 复制法向图
    copy_normals(args, images_name_list)
    args.normals = f"{args.cache_dir}/normals" if len(args.normals) > 0 else ""
    
def clear_cache(args):
    if not args.do_cache:
        return
    try:  
        shutil.rmtree(args.cache_dir)  
        print(f'{args.cache_dir} already delete') 
    except OSError as e:  
        print(f'{args.cache_dir} error delete {e.strerror}')

def copy_bbox_files(args):
    try:
        # Attempt to copy the file
        shutil.copy(f"{args.source_path}/center.txt", f"{args.model_path}/center.txt")
        shutil.copy(f"{args.source_path}/extent.txt", f"{args.model_path}/extent.txt")
    except Exception as e:
        # Catch other possible exceptions
        print(f"Warning: An error occurred while copying the file. Error message: {e}")


if __name__ == "__main__":
    # 打包时需要加入下面这句，否则 pyinstaller 打包后的 exe 遇到多进程语句就会重新初始化无法正确读入参数
    # 使用py脚本运行则需要注释这句
    torch.multiprocessing.freeze_support()

    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--auto_iteration', action="store_true", default=False) #
    parser.add_argument('--train_strength', type=float, default=1.0) # 训练强度，1为默认，2就代表训练2倍的轮数
    parser.add_argument('--remove_big_gs', action="store_true", default=False)

    parser.add_argument('--torch_ssim', dest='fused_ssim', action="store_false")
    parser.add_argument('--output_milestone', action="store_true", default=False)
    parser.add_argument('--skip_exist', action="store_true", default=False)
    parser.add_argument('--do_cache', action="store_true", default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default="D:/gs_cache")   # 当前cache_file的路径   默认下面有images depths两个文件夹  默认路径 f"D:\\gs_cache\\{args.gpu_id}" 
    parser.add_argument('--copy_sfm', action="store_true", default=True) # 可选是否复制colmap
    parser.add_argument('--layer_name', type=str, default="")   # 当前layer
    parser.add_argument('--chunk_name', type=str, default="")   # 当前chunk


    args, unknown_args = parser.parse_known_args(sys.argv[1:])
    
    #把所有路径处理一次
    args.cache_dir = os.path.normpath(args.cache_dir) if len(args.cache_dir) > 0 else args.cache_dir
    args.source_path = os.path.normpath(args.source_path) if len(args.source_path) > 0 else args.source_path
    args.images = os.path.normpath(args.images) if len(args.images) > 0 else args.images
    args.depths = os.path.normpath(args.depths) if len(args.depths) > 0 else args.depths
    args.model_path = os.path.normpath(args.model_path) if len(args.model_path) > 0 else args.model_path
    args.bounds_file = os.path.normpath(args.bounds_file) if len(args.bounds_file) > 0 else args.bounds_file
    args.scaffold_file = os.path.normpath(args.scaffold_file) if len(args.scaffold_file) > 0 else args.scaffold_file
    args.normals = os.path.normpath(args.normals) if len(args.normals) > 0 else args.normals

    if len(args.layer_name) > 0 and len(args.chunk_name) > 0:
        args.model_path = os.path.join(args.model_path, args.layer_name, args.chunk_name)

    #TODO need detect all iterations
    if args.skip_exist:
        if os.path.exists(f"{args.model_path}/point_cloud/iteration_final/point_cloud.ply"):
            print("Result already exits. skip. ")
            sys.exit(0)
  
    #自动设置参数
    if args.auto_iteration:
        print("Computing parameters.")
        training_image_num = get_train_image_num(args)
        factor = 20
        args.iterations = 10_000 + int(training_image_num * factor * args.train_strength)
        args.position_lr_max_steps = args.iterations
        args.densification_interval = 300#math.ceil(training_image_num/2)
        args.densify_from_iter = 2*training_image_num
        args.densify_until_iter = args.iterations

        stage_len = (args.densify_until_iter - args.densify_from_iter)//3
        args.stage_iterations = [args.densify_from_iter+stage_len, args.densify_from_iter+2*stage_len, args.densify_until_iter]

        args.opacity_reset_iters = [args.densify_from_iter, args.densify_from_iter+stage_len+1, args.densify_from_iter+2*stage_len+1] 
        args.opacity_remove_iter = [args.densify_from_iter+stage_len-1, args.densify_from_iter+2*stage_len-1]
        #print(f"stage iterations: {args.stage_iterations}")
        #print(f"opacity reset iters: {args.opacity_reset_iters}")
        #print(f"opacity remove iters: {args.opacity_remove_iter}")
        if args.remove_big_gs:
            args.remove_big_gs_iter = [i for i in range(args.densify_from_iter, args.densify_from_iter+2*stage_len, int(4*args.densification_interval))] 
        args.sphere_to_ellipsoid_iter = args.densify_from_iter+2*training_image_num if args.use_sphere else -1
        

        # 确定加入的步数之后添加
        args.regularization_from_iter = -1#int(args.iterations * 0.3) if args.regularization_from_iter == -1 else args.regularization_from_iter
        args.normal_from_iter = int(args.iterations * 0.3) if len(args.normals) else -1
        args.position_lr_init = 0.00016
        args.position_lr_final = 0.0000016
        args.max_gs_num = min(args.max_gs_num, 2*args.max_gs_num_in_box)+ args.skybox_num
      
        # 打印上面的参数args
        print(f"Training image num: {training_image_num}")
        print(f"Densify from iter: {args.densify_from_iter}")
        print(f"Densify until iter: {args.densify_until_iter}")
        print(f"Sphere to ellipsoid iter: {args.sphere_to_ellipsoid_iter}")
        print(f"Opacity reset iters: {args.opacity_reset_iters}")
        print(f"Opacity remove iter: {args.opacity_remove_iter}")
        print(f"Position lr init: {args.position_lr_init}")
        print(f"Position lr final: {args.position_lr_final}")
        print(f"Max gs num: {args.max_gs_num}")
       

    else:
        args.opacity_reset_iters = [args.opacity_reset_iters[0]]
        if len(args.stage_iterations) < 3:
            # 最小训练模式下允许不显式提供阶段切换点，避免直接索引空列表导致训练提前失败。
            args.stage_iterations = [args.densify_until_iter, args.densify_until_iter, args.densify_until_iter]
       

    if args.fused_ssim and fused_ssim is None:
        print("[train_single] fused_ssim 不可用，自动回退到 torch ssim。")
        args.fused_ssim = False

    print("Optimizing " + args.model_path)


    if args.eval and args.exposure_lr_init > 0 and not args.train_test_exp: 
        print("Reconstructing for evaluation (--eval) with exposure optimization on the train set but not for the test set.")
        print("This will lead to high error when computing metrics. To optimize exposure on the left half of the test images, use --train_test_exp")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print("caching.")
    cache_train_data(args)

    log_dir = f"{args.model_path}/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    # Start training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.output_milestone, writer)
    writer.close()
    copy_bbox_files(args)
    print("deleting cache.")
    clear_cache(args)
    # All done
    print("\nTraining complete.")
