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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._exp_name = ""
        self._images = ""
        self._alpha_masks = ""
        self._depths = ""
        self._normals = "" 
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False # Include the left half of the test images in the train set to optimize exposures
        self.data_device = "disk"
        self.eval = False
        self.skip_scale_big_gauss = False
        self.hierarchy = ""
        self.pretrained = ""
        self.skybox_num = 0
        self.scaffold_file = ""
        self.bounds_file = ""
        self.skybox_locked = False
        # 是否使用球进行训练  转换为椭球的迭代步在OptimizationParams 中给定
        self.use_sphere = False   
        # 从一个初始高斯开始，进一步优化
        self.init_gs_path = ""
        self.progress_path = ""    # 生成progress的文件夹
       
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.fused_ssim = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        #主要作用是控制显存占用
        self.max_gs_num = 2_000_0000
        #主要作用是为了控制LOD的分辨率
        self.max_gs_num_in_box = 1_000_0000
        self.iterations = 30_000
        self.position_lr_init = 0.00002
        self.position_lr_final = 0.0000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.001
        self.exposure_lr_final = 0.0001
        self.exposure_lr_delay_steps = 5000
        self.exposure_lr_delay_mult = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_iters = [i for i in range(500, 30_000, 3000)]
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        #self.densify_grad_threshold = 0.015
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01


        # 球变成椭球的间隔 对应ModelParams的use_sphere
        self.sphere_to_ellipsoid_iter = 15_000   
        self.remove_big_gs_iter = []
        # 包围壳之外的点符合分裂条件后分裂的比例  1 全部保存  0 全部删除 (需要输入包围壳的bounds_file)
        self.percent_outside = 1.        

        # 法向约束
        self.normal_from_iter = -1

        # Rade_Gs
        self.regularization_from_iter = -1    # 不使用
        self.lambda_depth_normal = 0.05

        self.opacity_remove_iter = []

        # 每个阶段的结束迭代位置
        self.stage_iterations = []
        # 法向约束
        self.normal_from_iter = -1

        # Rade_Gs
        self.regularization_from_iter = -1    # 不使用
        self.lambda_depth_normal = 0.05

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
