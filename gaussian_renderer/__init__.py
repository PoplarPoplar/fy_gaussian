# 功能：统一封装高斯渲染接口，兼容当前环境中不同版本的 diff_gaussian_rasterization 签名。
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import os
import sys
import importlib.util
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


RASTER_SETTING_FIELDS = getattr(GaussianRasterizationSettings, "_fields", tuple())
HIERARCHY_RASTERIZER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".third_party",
    "hierarchy_rasterizer",
)
_hierarchy_rasterizer_module = None


def _load_hierarchy_rasterizer():
    global _hierarchy_rasterizer_module
    if _hierarchy_rasterizer_module is not None:
        return _hierarchy_rasterizer_module

    package_dir = os.path.join(HIERARCHY_RASTERIZER_DIR, "diff_gaussian_rasterization")
    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.exists(init_path):
        raise FileNotFoundError(
            f"Hierarchy rasterizer package not found: {init_path}. "
            "Please build/install submodules/hierarchy-rasterizer into .third_party/hierarchy_rasterizer first."
        )

    spec = importlib.util.spec_from_file_location(
        "hierarchy_diff_gaussian_rasterization",
        init_path,
        submodule_search_locations=[package_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _hierarchy_rasterizer_module = module
    return module


def _build_raster_settings(
    viewpoint_camera,
    bg_color,
    scaling_modifier,
    sh_degree,
    pipe,
    kernel_size,
    require_coord,
    require_depth,
    require_max_weight_indices,
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if "kernel_size" in RASTER_SETTING_FIELDS:
        return GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            require_depth=require_depth,
            require_coord=require_coord,
            require_max_weight_indices=require_max_weight_indices,
            debug=pipe.debug,
        )

    if "depth_threshold" in RASTER_SETTING_FIELDS:
        empty_long = torch.empty(0, dtype=torch.int32, device="cuda")
        empty_float = torch.empty(0, dtype=torch.float32, device="cuda")
        return GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            depth_threshold=0.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            render_indices=empty_long,
            parent_indices=empty_long,
            interpolation_weights=empty_float,
            num_node_kids=empty_long,
            do_depth=require_depth,
        )

    raise RuntimeError(f"Unsupported GaussianRasterizationSettings fields: {RASTER_SETTING_FIELDS}")


def _normalize_raster_outputs(raw_outputs, viewpoint_camera, require_depth):
    if "kernel_size" in RASTER_SETTING_FIELDS:
        rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal, maxweight_indices = raw_outputs
        return (
            rendered_image,
            radii,
            rendered_expected_coord,
            rendered_median_coord,
            rendered_expected_depth,
            rendered_median_depth,
            rendered_alpha,
            rendered_normal,
            maxweight_indices,
        )

    if "depth_threshold" in RASTER_SETTING_FIELDS:
        rendered_image, radii, rendered_alpha, rendered_depth, maxweight_indices = raw_outputs
        image_height = int(viewpoint_camera.image_height)
        image_width = int(viewpoint_camera.image_width)
        rendered_expected_coord = torch.empty(0, device=rendered_image.device)
        rendered_median_coord = torch.empty(0, device=rendered_image.device)
        rendered_expected_depth = rendered_depth if require_depth else torch.empty(0, device=rendered_image.device)
        rendered_median_depth = rendered_depth if require_depth else torch.empty(0, device=rendered_image.device)
        rendered_normal = torch.zeros((3, image_height, image_width), dtype=rendered_image.dtype, device=rendered_image.device)
        return (
            rendered_image,
            radii,
            rendered_expected_coord,
            rendered_median_coord,
            rendered_expected_depth,
            rendered_median_depth,
            rendered_alpha,
            rendered_normal,
            maxweight_indices,
        )

    raise RuntimeError(f"Unsupported Gaussian rasterizer outputs for fields: {RASTER_SETTING_FIELDS}")


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size=0.0, scaling_modifier = 1.0, require_coord : bool = True, require_depth : bool = True, require_max_weight_indices : bool = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = _build_raster_settings(
        viewpoint_camera=viewpoint_camera,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        sh_degree=pc.active_sh_degree,
        pipe=pipe,
        kernel_size=kernel_size,
        require_coord=require_coord,
        require_depth=require_depth,
        require_max_weight_indices=require_max_weight_indices,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    opacity = pc.get_opacity
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    raw_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal, maxweight_indices = _normalize_raster_outputs(
        raw_outputs,
        viewpoint_camera,
        require_depth,
    )



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "normal":rendered_normal,
            "maxweight_indices": maxweight_indices
            }


def render_post(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    render_indices=None,
    parent_indices=None,
    interpolation_weights=None,
    num_node_kids=None,
    use_trained_exp=False,
    depth_threshold=None,
):
    """
    为 train_post/render_hierarchy 提供独立层级渲染入口。
    这里显式加载 .third_party/hierarchy_rasterizer 中构建出的层级 rasterizer，
    避免覆盖 train_single 使用的普通 diff_gaussian_rasterization。
    """

    hierarchy_rasterizer = _load_hierarchy_rasterizer()
    hierarchy_settings_cls = hierarchy_rasterizer.GaussianRasterizationSettings
    hierarchy_rasterizer_cls = hierarchy_rasterizer.GaussianRasterizer

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if render_indices is None:
        render_indices = torch.empty(0, dtype=torch.int32, device="cuda")
    if parent_indices is None:
        parent_indices = torch.empty(0, dtype=torch.int32, device="cuda")
    if interpolation_weights is None:
        interpolation_weights = torch.empty(0, dtype=torch.float32, device="cuda")
    if num_node_kids is None:
        num_node_kids = torch.empty(0, dtype=torch.int32, device="cuda")

    raster_settings = hierarchy_settings_cls(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold if depth_threshold is not None else -1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_kids,
        do_depth=True,
    )

    rasterizer = hierarchy_rasterizer_cls(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, _pixels, _depth_image, _maxweight_indices = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if use_trained_exp and hasattr(pc, "_exposure") and hasattr(pc, "exposure_mapping"):
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )
    rendered_image = rendered_image.clamp(0, 1)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
    }

# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate(points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size : float, scaling_modifier = 1.0, override_color = None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
        require_coord=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, point_coordinate, point_sdf, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "point_coordinate": point_coordinate,
            "point_sdf": point_sdf,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii}
