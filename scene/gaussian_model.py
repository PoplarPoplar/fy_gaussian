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

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy
from scene.OurAdam import Adam

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize




    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.nodes = None
        self.boxes = None

        self.writer = None
        self.iteration = 0
        self.pretrained_exposures = None

        self.skybox_points = 0
        self.skybox_locked = True

        self.use_sphere = False
        self.bbox_min = None
        self.bbox_max = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) if not self.use_sphere else self.scaling_activation(self._scaling[:, 0:1]).repeat(1, 3)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]
        # return self._exposure

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_writer(self, writer):
        self.writer = writer
    def set_iteration(self, iteration):
        self.iteration = iteration

    def create_from_pcd(
            self, 
            pcd : BasicPointCloud, 
            cam_infos : int,
            spatial_lr_scale : float,
            skybox_points: int,
            scaffold_file: str,
            bounds_file: str,
            skybox_locked: bool,
            use_sphere):
        
        self.spatial_lr_scale = spatial_lr_scale
        self.use_sphere = use_sphere
        # 记录初始点云数量，当球形高斯多于N(=2)倍初始点云数量时，停止增加球形高斯
        self.init_point_num = len(pcd.points)

        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked
        if scaffold_file != "" and skybox_points > 0:
            print(f"Overriding skybox_points: loading skybox from scaffold_file: {scaffold_file}")
            skybox_points = 0
        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean).cpu()    # 天空球半径

            # 20240831b golden angle in radians  生成更均匀的球  可能会在地面出现空洞保留原版
            # samples = torch.arange(self.skybox_points).float().cpu()
            # phi = torch.pi * (torch.sqrt(torch.tensor([5.])) - 1.)
            # y = 1 - (samples / (self.skybox_points - 1)) * 2  # [-1, 1]
            # r = torch.sqrt(1 - y * y)         # 计算中间结果
            # theta = phi * samples
            # x = torch.cos(theta) * r
            # z = torch.sin(theta) * r

            # skybox_xyz = torch.zeros((self.skybox_points, 3))
            # skybox_xyz[:, 0] = radius * 10 * x
            # skybox_xyz[:, 1] = radius * 10 * y
            # skybox_xyz[:, 2] = radius * 10 * z
            # skybox_xyz += mean.cpu()
            # xyz = torch.concat((skybox_xyz.cuda(), xyz))
            # fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            # fused_color[:self.skybox_points,0] *= 0.7
            # fused_color[:self.skybox_points,1] *= 0.8
            # fused_color[:self.skybox_points,2] *= 0.95

            # 原版层级Gauss生成的球 不包含地面方向上的点
            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            xyz = torch.concat((skybox_xyz.cuda(), xyz))
            fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            fused_color[:skybox_points,0] *= 0.7
            fused_color[:skybox_points,1] *= 0.8
            fused_color[:skybox_points,2] *= 0.95

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        if scaffold_file == "" and skybox_points > 0:
            dist2[:skybox_points] *= 10
            dist2[:skybox_points] = torch.clamp_max(dist2[:skybox_points], 6e4)      # 为了一些渲染器如supersplat对于scale较大会有不显示的问题
            dist2[skybox_points:] = torch.clamp_max(dist2[skybox_points:], 10) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # #改成统一大小，0.2米
        # scales = torch.log(0.2 * torch.ones((xyz.shape[0], 3), dtype=torch.float, device="cuda"))
        # # 前面10w个点为天空盒，给一个较大的scale
        # scales[:skybox_points] = torch.log(10 * torch.ones((skybox_points, 3), dtype=torch.float, device="cuda"))
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if scaffold_file == "" and skybox_points > 0:
            opacities = self.inverse_opacity_activation(0.2 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_points] = 0.7
        else: 
            opacities = self.inverse_opacity_activation(0.01 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self.scaffold_points = None
        if scaffold_file != "": 
            scaffold_sh_degree = 1     # scaffold 的 sh_degree  train_coarse中产生, 默认为1 
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", scaffold_sh_degree)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                skybox_points = int(f.readline())

            self.skybox_points = skybox_points
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()
                    self.bbox_min = center - extent / 2
                    self.bbox_max = center + extent / 2

            distances1 = torch.abs(scaffold_xyz.cuda() - center)
            selec = torch.logical_and(
                torch.max(distances1[:,0], distances1[:,1]) > 0.5 * extent[0],
                torch.max(distances1[:,0], distances1[:,1]) < 1.5 * extent[0])
            selec[:skybox_points] = True

            self.scaffold_points = selec.nonzero().size(0)

            xyz = torch.concat((scaffold_xyz.cuda()[selec], xyz))
            features_dc = torch.concat((features_dc_scaffold.cuda()[selec,0:1,:], features_dc))

            # 根据 max_sh_degree 进行更改了源码
            rest_size = (self.max_sh_degree + 1) ** 2 - 1                                                           # 应该保存的大小
            scaffold_use_size = (min(self.max_sh_degree, scaffold_sh_degree) + 1) ** 2 - 1                          # scaffold 读取后可以利用的大小
            filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), rest_size, 3))      
            filler[:, 0:scaffold_use_size, :] = features_extra_scaffold.cuda()[selec, 0:scaffold_use_size, :]       # 只保留脚手架中与sh             
            features_rest = torch.concat((filler.cuda(), features_rest))
            scales = torch.concat((scales_scaffold.cuda()[selec], scales))
            rots = torch.concat((rots_scaffold.cuda()[selec], rots))
            opacities = torch.concat((opacities_scaffold.cuda()[selec], opacities))
        
        elif bounds_file != "":
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()
                    # extent 外扩0.1
                    extent = extent * 1.1
                    self.bbox_min = center - extent / 2
                    self.bbox_max = center + extent / 2

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        print("Number of points at initialisation : ", self._xyz.shape[0])

    def training_setup(self, training_args, our_adam=True):
        self.percent_dense = training_args.percent_dense
        self.percent_outside = training_args.percent_outside
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if our_adam:
            self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)

       
    def load_ply_file(self, path, degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        #assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
        if len(extra_f_names) == 0:
            features_extra = np.zeros((xyz.shape[0], 3*(degree + 1) ** 2 - 3))
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))
        else:
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots


    def create_from_init_gs(self, gs_path, bounds_file, cam_infos, spatial_lr_scale, skybox_points, skybox_locked, use_sphere):
        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.scaffold_points = None
        if not os.path.exists(gs_path):
            raise Exception(f"No file found at {gs_path}")
       
       
        if len(bounds_file):
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()
                    bbox_expan_factor = 1.1
                    self.bbox_min = center - (extent*bbox_expan_factor) / 2
                    self.bbox_max = center + (extent*bbox_expan_factor) / 2


        # 高斯文件
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_ply_file(gs_path, self.max_sh_degree)

        xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda")
        features_extra = torch.tensor(features_extra, dtype=torch.float, device="cuda")
        opacities = torch.tensor(opacities, dtype=torch.float, device="cuda")
        scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        # opacities*= 0.5
        scales=torch.log(torch.exp(scales)*4)
        rots = torch.tensor(rots, dtype=torch.float, device="cuda")

        # 天空球
        self.use_sphere = use_sphere
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked

        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean).cpu()    # 天空球半径

            # 20240831b golden angle in radians  生成更均匀的球  可能会在地面出现空洞保留原版
            # samples = torch.arange(self.skybox_points).float().cpu()
            # phi = torch.pi * (torch.sqrt(torch.tensor([5.])) - 1.)
            # y = 1 - (samples / (self.skybox_points - 1)) * 2  # [-1, 1]
            # r = torch.sqrt(1 - y * y)         # 计算中间结果
            # theta = phi * samples
            # x = torch.cos(theta) * r
            # z = torch.sin(theta) * r

            # skybox_xyz = torch.zeros((self.skybox_points, 3))
            # skybox_xyz[:, 0] = radius * 10 * x
            # skybox_xyz[:, 1] = radius * 10 * y
            # skybox_xyz[:, 2] = radius * 10 * z
            # skybox_xyz += mean.cpu()
            # xyz = torch.concat((skybox_xyz.cuda(), xyz))
            # fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            # fused_color[:self.skybox_points,0] *= 0.7
            # fused_color[:self.skybox_points,1] *= 0.8
            # fused_color[:self.skybox_points,2] *= 0.95

            # 原版层级Gauss生成的球 不包含地面方向上的点
            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            skybox_xyz = skybox_xyz.cuda()
            xyz = torch.concat((skybox_xyz, xyz))

            #天空颜色，蓝色
            skybox_color = torch.ones((skybox_points, 3)).cuda()
            skybox_color[:skybox_points,0] *= 0.7
            skybox_color[:skybox_points,1] *= 0.8
            skybox_color[:skybox_points,2] *= 0.95
            skybox_features = torch.zeros((skybox_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            skybox_features[:, :3, 0 ] = RGB2SH(skybox_color)
            skybox_features[:, 3:, 1:] = 0.0
            features_dc = torch.concat((skybox_features[:,:,0:1], features_dc))
            features_extra = torch.concat((skybox_features[:,:,1:], features_extra))
        
            # 天空球scale
            dist2 = torch.clamp_min(distCUDA2(skybox_xyz), 0.0000001)
       
            dist2 *= 10
            dist2 = torch.clamp_max(dist2, 6e4)      # 为了一些渲染器如supersplat对于scale较大会有不显示的问题
            skybox_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            scales = torch.concat((skybox_scales.cuda(), scales))

            #天空球 rot
            skybox_rots = torch.zeros((skybox_xyz.shape[0], 4), device="cuda")
            skybox_rots[:, 0] = 1
            rots = torch.concat((skybox_rots.cuda(), rots))

            #不透明度
            skybox_opacities =0.7 * torch.ones((skybox_xyz.shape[0], 1), dtype=torch.float, device="cuda")
            opacities = torch.concat((skybox_opacities.cuda(), opacities))
        
            self.scaffold_points = None

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_extra.transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        


    def create_from_hier(self, path, spatial_lr_scale : float, scaffold_file : str):
        self.spatial_lr_scale = spatial_lr_scale

        xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)

        base = os.path.dirname(path)

        try:
            with open(os.path.join(base, "anchors.bin"), mode='rb') as f:
                bytes = f.read()
                int_val = int.from_bytes(bytes[:4], "little", signed="False")
                dt = np.dtype(np.int32)
                vals = np.frombuffer(bytes[4:], dtype=dt) 
                self.anchors = torch.from_numpy(vals).long().cuda()
        except:
            print("WARNING: NO ANCHORS FOUND")
            self.anchors = torch.Tensor([]).long()

        #retrieve exposure
        exposure_file = os.path.join(base, "exposure.json")
        if os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)

            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        else:
            print(f"No exposure to be loaded at {exposure_file}")
            self.pretrained_exposures = None

        #retrieve skybox
        self.skybox_points = 0         
        if scaffold_file != "":
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                    skybox_points = int(f.readline())

            self.skybox_points = skybox_points

        if self.skybox_points > 0:
            if scaffold_file != "":
                skybox_xyz, features_dc_sky, features_rest_sky, opacities_sky, scales_sky, rots_sky = scaffold_xyz[:skybox_points], features_dc_scaffold[:skybox_points], features_extra_scaffold[:skybox_points], opacities_scaffold[:skybox_points], scales_scaffold[:skybox_points], rots_scaffold[:skybox_points]

            opacities_sky = torch.sigmoid(opacities_sky)
            xyz = torch.cat((xyz, skybox_xyz))
            alpha = torch.cat((alpha, opacities_sky))
            scales = torch.cat((scales, scales_sky))
            rots = torch.cat((rots, rots_sky))
            filler = torch.zeros(features_dc_sky.size(0), 16, 3)
            filler[:, :1, :] = features_dc_sky
            filler[:, 1:4, :] = features_rest_sky
            shs_all = torch.cat((shs_all, filler))

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = torch.abs

        self.hierarchy_path = path

        self.nodes = nodes.cuda()
        self.boxes = boxes.cuda()

    def create_from_pt(self, path, spatial_lr_scale : float ):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.load(path + "/done_xyz.pt")
        shs_dc = torch.load(path + "/done_dc.pt")
        shs_rest = torch.load(path + "/done_rest.pt")
        alpha = torch.load(path + "/done_opacity.pt")
        scales = torch.load(path + "/done_scaling.pt")
        rots = torch.load(path + "/done_rotation.pt")

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_dc.cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_rest.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_hier(self):
        write_hierarchy(self.hierarchy_path + "_opt",
                        self._xyz,
                        torch.cat((self._features_dc, self._features_rest), 1),
                        self.opacity_activation(self._opacity),
                        self._scaling,
                        self._rotation,
                        self.nodes,
                        self.boxes)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_pt(self, path):
        mkdir_p(path)

        torch.save(self._xyz.detach().cpu(), os.path.join(path, "done_xyz.pt"))
        torch.save(self._features_dc.cpu(), os.path.join(path, "done_dc.pt"))
        torch.save(self._features_rest.cpu(), os.path.join(path, "done_rest.pt"))
        torch.save(self._opacity.cpu(), os.path.join(path, "done_opacity.pt"))
        torch.save(self._scaling, os.path.join(path, "done_scaling.pt"))
        torch.save(self._rotation, os.path.join(path, "done_rotation.pt"))

        import struct
        def load_pt(path):
            xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
            features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
            features_rest = torch.load( os.path.join(path, "done_rest.pt")).detach().cpu()
            opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
            scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
            rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()

            return xyz, features_dc, features_rest, opacity, scaling, rotation


        xyz, features_dc, features_rest, opacity, scaling, rotation = load_pt(path)

        my_int = xyz.size(0)
        with open(os.path.join(path, "point_cloud.bin"), 'wb') as f:
            f.write(struct.pack('i', my_int))
            f.write(xyz.numpy().tobytes())
            print(features_dc[0])
            print(features_rest[0])
            f.write(torch.cat((features_dc, features_rest), dim=1).numpy().tobytes())
            f.write(opacity.numpy().tobytes())
            f.write(scaling.numpy().tobytes())
            f.write(rotation.numpy().tobytes())


    def save_ply(self, path, without_sky = True, bbox_only = True):
        mkdir_p(os.path.dirname(path))
        save_mask = torch.ones(self._xyz.shape[0], dtype=bool, device=self.get_xyz.device)
        if without_sky:
            save_mask[:self.skybox_points] = False
        

        if bbox_only and self.bbox_max is not None:
            inside_bbox_x = torch.logical_and(self.get_xyz[:, 0] > self.bbox_min[0], self.get_xyz[:, 0] < self.bbox_max[0])
            inside_bbox_y = torch.logical_and(self.get_xyz[:, 1]  > self.bbox_min[1], self.get_xyz[:, 1]  < self.bbox_max[1])
            inside_bbox_z = torch.logical_and(self.get_xyz[:, 2]  > self.bbox_min[2], self.get_xyz[:, 2]  < self.bbox_max[2])  
            inside_bbox = torch.logical_and(inside_bbox_x, torch.logical_and(inside_bbox_y, inside_bbox_z))
        
            save_mask = torch.logical_and(save_mask, inside_bbox)

        # 跳过前 skybox_point_num 个点
        xyz = self._xyz.detach()[save_mask].cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[save_mask].cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[save_mask].cpu().numpy()
        opacities = self._opacity.detach()[save_mask].cpu().numpy()
        scale = self._scaling[:, 0:1].repeat(1, 3).detach()[save_mask].cpu().numpy() if self.use_sphere else self._scaling.detach()[save_mask].cpu().numpy()
        rotation = self._rotation.detach()[save_mask].cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_opacity[self.skybox_points:], torch.ones_like(self.get_opacity[self.skybox_points:])*0.01))), 0)
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_ply_file(path, self.max_sh_degree)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")     # 重新加载
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def get_outside_gs_num(self):
        "得到所有在边界外的点数（不包括天空）"
        if self.bbox_max is None:
            return 0
        outside_bbox = ~torch.all((self.get_xyz[self.skybox_points:] >= self.bbox_min) & 
                (self.get_xyz[self.skybox_points:] <= self.bbox_max), dim=1)

        return torch.sum(outside_bbox[self.skybox_points:]).item()
    def simplify_points(self, target_num_in_box):
        """将包围盒内的点云随机采样到num_in_box数量
        包围盒外的点云不进行采样
        """
        if self.bbox_max is None:
            # 如果没有包围盒，就把除了天空球点以外的点随机采样到target_num_in_box数量
            # 要删除的点的数量
            to_delete_num  = self.get_xyz[self.skybox_points:].shape[0] - target_num_in_box
            if to_delete_num > 0:
                delete_index = torch.randperm(self.get_xyz[self.skybox_points:].shape[0])[:to_delete_num]
                delete_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool)
                delete_mask[self.skybox_points:][delete_index] = True
                self.prune_points(delete_mask)
        else:
            # 判断哪些点在包围盒内，排除天空盒上的点
            inside_bbox = torch.all(
                (self.get_xyz[self.skybox_points:] >= self.bbox_min) & 
                (self.get_xyz[self.skybox_points:] <= self.bbox_max), dim=1
            )

            inbbox_point_num = inside_bbox.sum().item()

            # 如果包围盒内的点数小于目标值，直接返回
            num_to_delete_inside = inbbox_point_num - target_num_in_box
            if num_to_delete_inside <= 0:
                return

            # 如果包围盒内的点数大于目标值，随机采样要删除的点
            inside_index = torch.nonzero(inside_bbox).squeeze(1)
            delete_index = inside_index[torch.randperm(inbbox_point_num)[:num_to_delete_inside]]
            delete_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool)
            delete_mask[self.skybox_points:][delete_index] = True
          
           
            self.prune_points(delete_mask)


    def simplify_points2(self, target_num_in_box, input_delete_mask):
        """将包围盒内的点云随机采样到num_in_box数量，input_delete_mask是要优先删除的点
         一共是3种情况
         1如果包围盒内的点已经小于target_num_in_box了，那么就不要删除了,这种情况比较难办，又不能在这里增加一些点，所以就不处理了
         2如果按照input_delete_mask去删除点，删除完了包围盒内小于等于target_num_in_box，那么就不能按input_delete_mask删除，要随机减少input_delete_mask中的为true的数量，使得包围盒达到target_num_in_box
         3如果按照input_delete_mask去删除点，删除完了包围盒大于target_num_in_box，那么就随机增加input_delete_mask中的为true的数量，特别要注意天空球对应的点，使得包围盒达到target_num_in_box
        """
        # 禁止删除天空盒上的点
        input_delete_mask[:self.skybox_points] = False

        if self.bbox_max is None:
            # 如果包围盒不存在，则处理整个点云
             # 要删除的点的数量
            to_delete_num  = self.get_xyz[self.skybox_points:].shape[0] - target_num_in_box
            if to_delete_num <= 0:
                return
            
            # 需要删除更多的点吗？
            more_delete_num =  to_delete_num - input_delete_mask[self.skybox_points:].sum().item()
            if more_delete_num > 0:
                # 如果已经标记的点小于要删除的点，先找找到input_delete_mask为false的点，随机选择to_delete_num - input_delete_mask[self.skybox_points:].sum()个
                not_delete_indices = torch.nonzero(~input_delete_mask[self.skybox_points:]).squeeze(-1)
                not_delete_indices = not_delete_indices[torch.randperm(len(not_delete_indices))][:more_delete_num]
                input_delete_mask[self.skybox_points:][not_delete_indices] = True
            else:
                # 如果已经标记和点大于于要删除的点，先找到input_delete_mask为true的点，从中恢复 -more_delete_num个点
                to_delete_indices = torch.nonzero(input_delete_mask[self.skybox_points:]).squeeze(-1)
                recover_indices = to_delete_indices[torch.randperm(len(to_delete_indices))][:-more_delete_num]
                input_delete_mask[self.skybox_points:][recover_indices] = False
            self.prune_points(input_delete_mask)
        else:
            # 如果包围盒存在，则只处理包围盒内的点
            # 只处理非天空盒的点
            xyz_without_skybox = self.get_xyz[self.skybox_points:]
            # 判断这些点是否在包围盒内
            inside_bbox = torch.all(
                (xyz_without_skybox >= self.bbox_min) & (xyz_without_skybox <= self.bbox_max), dim=1
            )
            inbbox_point_num = inside_bbox.sum().item()

            if inbbox_point_num <= target_num_in_box:
                print(f'WARNING: point inside bbox = {inbbox_point_num} is less than target number in bbox = {target_num_in_box}, cannot delete points')
                return  # 第一种情况

            # 计算包围盒内的点删除后的数量
            inbbox_num_after_delete = inbbox_point_num - torch.logical_and(inside_bbox, input_delete_mask[self.skybox_points:]).sum().item()

            if inbbox_num_after_delete <= target_num_in_box:
                # 第二种情况：减少删除量，避免包围盒内的点数低于目标数量
                inbbox_delete_mask = torch.logical_and(inside_bbox, input_delete_mask[self.skybox_points:])
                outbbox_delete_mask = torch.logical_and(~inside_bbox, input_delete_mask[self.skybox_points:])

                inbbox_delete_num = inbbox_delete_mask.sum().item()
                outbbox_delete_num = outbbox_delete_mask.sum().item()

                # 计算要保留的点数
                inbbox_keep_num = target_num_in_box - inbbox_num_after_delete
                out_in_ratio = outbbox_delete_num / inbbox_delete_num
                outbbox_keep_num = int(inbbox_keep_num * out_in_ratio)

                # 随机选择要保留的包围盒内外点
                inbbox_delete_index = torch.nonzero(inbbox_delete_mask).squeeze(-1)
                inbbox_keep_index = inbbox_delete_index[torch.randperm(len(inbbox_delete_index))][:inbbox_keep_num]

                outbbox_delete_index = torch.nonzero(outbbox_delete_mask).squeeze(-1)
                outbbox_keep_index = outbbox_delete_index[torch.randperm(len(outbbox_delete_index))][:outbbox_keep_num]

                # 更新删除标记
                input_delete_mask[self.skybox_points:][inbbox_keep_index] = False
                input_delete_mask[self.skybox_points:][outbbox_keep_index] = False

            else:
                # 第三种情况：增加删除量，避免包围盒内的点数超过目标数量
                inbbox_not_delete_mask = torch.logical_and(inside_bbox, ~input_delete_mask[self.skybox_points:])
                outbbox_not_delete_mask = torch.logical_and(~inside_bbox, ~input_delete_mask[self.skybox_points:])

                inbbox_not_delete_num = inbbox_not_delete_mask.sum().item()
                outbbox_not_delete_num = outbbox_not_delete_mask.sum().item()

                out_in_ratio = outbbox_not_delete_num / inbbox_not_delete_num
                inbbox_extra_delete_num = -(target_num_in_box - inbbox_num_after_delete)
                outbbox_extra_delete_num = int(inbbox_extra_delete_num * out_in_ratio)

                # 随机选择要进一步删除的点
                inbbox_not_delete_index = torch.nonzero(inbbox_not_delete_mask).squeeze(-1)
                inbbox_extra_delete_index = inbbox_not_delete_index[torch.randperm(len(inbbox_not_delete_index))][:inbbox_extra_delete_num]

                outbbox_not_delete_index = torch.nonzero(outbbox_not_delete_mask).squeeze(-1)
                outbbox_extra_delete_index = outbbox_not_delete_index[torch.randperm(len(outbbox_not_delete_index))][:outbbox_extra_delete_num]

                # 更新删除标记
                input_delete_mask[self.skybox_points:][inbbox_extra_delete_index] = True
                input_delete_mask[self.skybox_points:][outbbox_extra_delete_index] = True

            # 禁止删除天空盒上的点
            input_delete_mask[:self.skybox_points] = False

            # 调用prune_points函数进行删除操作
            self.prune_points(input_delete_mask)




    def remove_big_scale(self):
        """
        根据 Gaussians 的最大轴向尺度，删除“尺度特别大”的点。
        如果定义了 bounding box (self.bbox_min, self.bbox_max)，
        只在 bbox 范围内的点上进行统计与删除；
        如果未定义 bounding box，则在全体点上进行统计与删除。
        """

        # 1. 获取所有点的最大轴向尺度
        #    (e.g. 若 self.get_scaling() 返回 [N, 3]，则取每行的 max)
        max_axis_scale_without_skybox = torch.max(self.get_scaling[self.skybox_points:], dim=1).values  # shape: [N]

        # 2. 获取所有点的位置
        xyz_without_skybox = self.get_xyz[self.skybox_points:]

        # 3. 确定哪些点属于 bounding box（inside_bbox）：
        #    若没有 bbox_min / bbox_max，则视为全部点都在范围内
        if (self.bbox_min is not None) and (self.bbox_max is not None):
            # 判断是否在包围盒之内
            # inside_bbox[i] = True 表示 positions[i] 在 [bbox_min, bbox_max] 范围内
             inside_bbox = torch.all(
                (xyz_without_skybox >= self.bbox_min) & (xyz_without_skybox <= self.bbox_max), dim=1
            )
        else:
            # 如果没有设置 bounding box，则全部点都参与
            inside_bbox = torch.ones_like(max_axis_scale_without_skybox, dtype=torch.bool)


        # 5. 如果没有“有效”的点参与阈值计算，直接退出
        if inside_bbox.sum() == 0:
            print("No points available for threshold calculation in bounding box.")
            return

        # 6. 计算阈值(示例：取 0.5 分位数 * 10)
        scale_threshold = 10.0 * torch.quantile(
            max_axis_scale_without_skybox[inside_bbox],
            0.5  # 你可以根据需求调整分位数
        )

        # 7. 生成删除掩码 remove_mask：
        #    - 大于阈值
        #    - 且在 bounding box 内
        remove_mask_without_skybox = (max_axis_scale_without_skybox > scale_threshold) & inside_bbox
        # 再加上 skybox 的点
        remove_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        remove_mask[self.skybox_points:][remove_mask_without_skybox] = True
        # 9. 打印信息并执行删除
        num_removed = remove_mask.sum()
        print(f"remove_top_percent_scale: {num_removed} points will be removed.")
        if num_removed > 0:
            self.prune_points(remove_mask)
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum =   torch.cat((self.xyz_gradient_accum, torch.zeros((new_xyz.shape[0], 1), device="cuda")))
        self.denom =  torch.cat((self.denom, torch.zeros((new_xyz.shape[0], 1), device="cuda")))
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_select_point(self, grads, grow_speed, strategy):
        n_init_points = self.get_xyz.shape[0]
        
        # -------------------------------------------------------------------------
        # 1. 准备好 grads 和 weighted_grad
        # -------------------------------------------------------------------------
        # 先对 grads 做 padding，避免越界
        padded_grad = torch.zeros(n_init_points, device="cuda")
        if grads.numel() > 0:
            padded_grad[:grads.shape[0]] = grads.squeeze()

        # 注意：只对非天空点做后续逻辑，因此 skybox_points 前面的点跳过
        # weighted_grad 的计算方式沿用之前的逻辑
        weighted_grad = padded_grad * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1 / 5.0)

        # -------------------------------------------------------------------------
        # 2. 构建“候选分裂点”掩码
        #    分为以下条件：
        #    - denom > 20
        #    - weighted_grad > grad_threshold
        # -------------------------------------------------------------------------

        candidate_mask  = (self.denom.squeeze(-1) >= 1)
        
        # 排除skybox_points
        candidate_mask[:self.skybox_points] = False
        candidate_indices = torch.where(candidate_mask)[0] 
        # 如果没有符合条件的候选点，就不做任何分裂，直接返回
        if candidate_indices.numel() == 0:
            return
        
        # -------------------------------------------------------------------------
        # 3. 在候选点里，根据 weighted_grad 由大到小排序
        # -------------------------------------------------------------------------
        # 把候选索引对应的 weighted_grad 拿出来排序
        candidate_wgrad = weighted_grad[candidate_indices]
        sorted_indices = torch.argsort(candidate_wgrad, descending=True)
        # 得到排好序的 candidate_indices（这里是相对于非天空点的下标）
        candidate_indices = candidate_indices[sorted_indices]
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        # 4. 决定本次要分裂多少点 select_num
        #    （以下逻辑可根据自己的需求调整，也可保留原策略）
        # -------------------------------------------------------------------------
        # 原先的逻辑是：select_num = int((全部点数 - skybox_points) * grow_speed)
        # 并根据 strategy 不同，乘以 4 或 16，再随机取 select_num
        select_num = int((self.get_xyz.shape[0] - self.skybox_points) * grow_speed)
        if select_num > (self.get_xyz.shape[0] - self.skybox_points):
            select_num = self.get_xyz.shape[0] - self.skybox_points
    
        # 根据 strategy 不同，初步扩张一下候选集数量
        if strategy == 0:
            # 例如先取前 select_num*4
            top_indices = candidate_indices[:select_num]  
        elif strategy == 1:
            # 例如先取前 select_num*16
            top_indices = candidate_indices[:select_num]
        else:  # strategy == 2
            # 直接全取
            top_indices = candidate_indices[:select_num]  

        # 最后再随机从 top_indices 中选出 select_num 个
        if top_indices.numel() > select_num:
            perm = torch.randperm(top_indices.shape[0], device=top_indices.device)
            top_indices = top_indices[perm[:select_num]]
        else:
            # 如果候选量小于 select_num，就直接全部取
            pass
        selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
        selected_pts_mask[top_indices] = True

        # outside bbox mask
        if self.bbox_max is not None:
            outside_bbox_x = torch.logical_or(self.get_xyz[:, 0] < self.bbox_min[0], self.get_xyz[:, 0] > self.bbox_max[0])
            outside_bbox_y = torch.logical_or(self.get_xyz[:, 1] < self.bbox_min[1], self.get_xyz[:, 1] > self.bbox_max[1])
            outside_bbox_z = torch.logical_or(self.get_xyz[:, 2] < self.bbox_min[2], self.get_xyz[:, 2] > self.bbox_max[2])   

            outside_bbox = torch.logical_or(outside_bbox_x, torch.logical_or(outside_bbox_y, outside_bbox_z))
            inside_bbox = torch.logical_not(outside_bbox)
            #只有strategy==0，则包围盒外的点才有可能被选中
            bbox_mask_valid = inside_bbox
            if strategy == 0:
                outside_valid = torch.logical_and(outside_bbox, torch.rand(self.get_xyz.shape[0]).cuda() < self.percent_outside)
                bbox_mask_valid = torch.logical_or(outside_valid, bbox_mask_valid)
            selected_pts_mask = torch.logical_and(selected_pts_mask, bbox_mask_valid)
        # 分裂过的点，那么把denom置为0
        # self.denom[selected_pts_mask] = 0
        # self.xyz_gradient_accum[selected_pts_mask] = 0
        # self.max_radii2D[selected_pts_mask] = 0    
        return selected_pts_mask
        
    def densify_and_split(self, grads, split_threshold, grow_speed, strategy, N=2):
        selected_pts_mask = self.densify_select_point(grads, grow_speed, strategy)
        if selected_pts_mask is None:
            return
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > split_threshold)
        if self.writer is not None:
            self.writer.add_scalar("Num/split", selected_pts_mask.sum().item(), self.iteration)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, split_threshold, grow_speed, strategy):
        selected_pts_mask = self.densify_select_point(grads, grow_speed, strategy)
        if selected_pts_mask is None:
            return
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= split_threshold)
        if self.writer is not None:
            self.writer.add_scalar("Num/clone", selected_pts_mask.sum().item(), self.iteration)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, min_opacity, split_threshold, grow_speed, strategy):
        grads = self.xyz_gradient_accum
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, split_threshold, grow_speed, strategy)
        self.densify_and_split(grads, split_threshold, grow_speed, strategy)
      

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.scaffold_points is not None:
            prune_mask[:self.scaffold_points] = False
        if self.skybox_points is not None:
            prune_mask[:self.skybox_points] = False
        self.prune_points(prune_mask)

        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
    


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] = torch.max(torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_filter])
        self.denom[update_filter] += 1
    
    def sphere_to_ellipsoid(self):
        if self.use_sphere:
            print("sphere_to_ellipsoid")
            scaling_new = self.scaling_inverse_activation(self.get_scaling)        # 保存scale, use_sphere为True
            self.use_sphere = False
            optimizable_tensors = self.replace_tensor_to_optimizer(
                scaling_new, "scaling"
            )
            self._scaling = optimizable_tensors["scaling"]