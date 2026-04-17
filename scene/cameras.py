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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2

from utils.general_utils import PILtoTorch

import torch
import torch.nn.functional as F

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, primx, primy, image, alpha_mask,
                 invdepthmap, normal_map,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
        else:
            #make an empty image
            #4, H, W
            print("[Warning] Empty Image")
            resized_image_rgb = torch.zeros((3, resolution[1], resolution[0]), dtype=torch.float32)
            gt_image = resized_image_rgb[:3, ...]
        if alpha_mask is not None:
            self.alpha_mask = PILtoTorch(alpha_mask, resolution)
        elif resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if self.alpha_mask is not None:
            self.original_image *= self.alpha_mask

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
            invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            invdepthmapScaled[invdepthmapScaled < 0] = 0
            if invdepthmapScaled.ndim != 2:
                invdepthmapScaled = invdepthmapScaled[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

            if self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap > 0)
            
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]: 
                self.depth_mask *= 0
            else:
                self.depth_reliable = True
        
        self.normal_map = None
        self.gray_image = None
        self.normal_mask = None
        if normal_map is not None:
            # 以下变换是按照stable normal的计算做逆变换
            self.normal_map = PILtoTorch(normal_map, resolution)   # [0, 1]
            self.normal_map = self.normal_map.clip(0, 1) * 2       # [0, 2]
            self.normal_map = self.normal_map - 1                  # [-1, 1]
        
            gray_image = image.convert('L')
            self.gray_image = PILtoTorch(gray_image, resolution).clamp(0.0, 1.0).to(self.data_device)
            self.normal_mask = generate_normal_mask(self.gray_image).to(self.data_device)        # mask
            self.normal_mask = dilate_mask(self.normal_mask.to(torch.uint8).squeeze()) 

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, primx = primx, primy=primy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

@torch.no_grad()
def generate_normal_mask(tensor, percent=0.5):
    '''
        通过边缘检测返回一个较为平坦区域的mask(图像梯度小)
    '''
    # 定义Sobel滤波器
    sobel_x = torch.tensor([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # 应用Sobel滤波器
    sobelx = F.conv2d(tensor, sobel_x, padding=1)
    sobely = F.conv2d(tensor, sobel_y, padding=1)

    # 计算梯度幅度
    sobel = torch.sqrt(sobelx**2 + sobely**2)

    n = tensor.numel()
    k = int(n * percent)     # 图像梯度较小的前k个数 进行约束
    value, _ = sobel.view(-1).kthvalue(k)
    return sobel <= value

@torch.no_grad()
def dilate_mask(mask, kernel_size=3):
    # 确保输入的mask是二值的（0和1）
    assert mask.dim() == 2, "Input mask should be 2D"
    
    # 将mask的形状调整为4D，以便进行卷积操作
    mask = mask.unsqueeze(0).unsqueeze(0)  # 变为形状 [1, 1, H, W]
    
    # 使用最大池化进行膨胀
    dilated_mask = F.max_pool2d(- mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    dilated_mask = - dilated_mask
    # 返回膨胀后的掩码，去掉多余的维度
    return dilated_mask.squeeze().byte()  # 转换回0-1掩码
