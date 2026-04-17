# 2024/12/13
#
# 读取 ImageSimulation.json 文件并生成渲染图片
# 使用方法

import json
import os
import torch
import torchvision
import numpy as np
import multiprocessing
from argparse import ArgumentParser

from gauss_render import render

from render_utils.cameras_utils import Camera
from render_utils.graphics_utils import focal2fov
from render_utils.Gaussian import GaussianModel
from render_arguments import PipelineParams, get_combined_args

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_camera_info(cameras_list):
    """
    读取 ImageSimulation.json 中相机信息
    cameras_list: list, 带有 ColmapView、FocalLength、ImageSize、Name、Position 以及 View 信息的列表
    """
    cameras_dic = {}
    for i in range(len(cameras_list)):
        image_info = cameras_list[i]
        
        view_data = image_info["View"]
        view_matrix = np.array(view_data).reshape(4, 4).T
        # R = view_matrix[:3, :3]
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        T = view_matrix[:3, 3]
        # camera_position = -np.dot(R.T, T)
        camera_position = np.array([0, 0, 0])
                
        focal = image_info["FocalLength"]
        weight, height = image_info["ImageSize"]
        FoVx = focal2fov(focal, weight)
        FoVy = focal2fov(focal, height)
        
        iamge_name = image_info["Name"]
        
        camera = Camera(R, camera_position, weight, height, FoVx, FoVy)
        cameras_dic[iamge_name] = camera
    
    return cameras_dic

def render_images(model_path, output_path, cams_info, pipeline):
    """
    model_path: str, 模型路径
    output_path: str, 图片保存路径
    cams_info: dict, 含有 cams 类的字典
    """
    with torch.no_grad():
        gaussian = GaussianModel(0)
        gaussian.load_ply(model_path)
        
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for cam_name, cam_info in cams_info.items():
            render_image = render(cam_info, gaussian, pipeline, background)["render"]
            render_image_path = os.path.join(output_path, cam_name)
            torchvision.utils.save_image(render_image, render_image_path)
        print("All done!")

if __name__ == "__main__":
    # 打包时需要加入下面这句，否则 pyinstaller 打包后的 exe 遇到多进程语句就会重新初始化无法正确读入参数
    # 使用py脚本运行则需要注释这句
    multiprocessing.freeze_support()

    # torch.multiprocessing.set_sharing_strategy('file_system')
    multiprocessing.set_start_method("spawn", force=True)
    parser = ArgumentParser(description="Image Generated")
    pipeline = PipelineParams(parser)
    
    parser.add_argument("-p", "--project_path", type=str, help="the project in", default="")
    parser.add_argument("-m", "--model_path", type=str, help="Gaussian model located in", default="")
    parser.add_argument("-i", "--ImageSimulation_path", type=str, help="ImageSimulation.json located in", default="")
    parser.add_argument("-o", "--output_path", type=str, help="output path for images", default="")
    args = parser.parse_args()
    
    # 读取 ImageSimulation_info.json 信息
    ImageSimulation_path = os.path.join(args.project_path, "ImageSimulation.json") if args.ImageSimulation_path == "" else args.ImageSimulation_path
    ImageSimulation_info = read_json_file(ImageSimulation_path)
    
    # 设置输出路径
    output_path = os.path.join(args.project_path, "ImageSimulation") if args.output_path == "" else args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    # 设置模型路径
    ModelPath = ImageSimulation_info.get("ModelPath")
    model_path = args.model_path or ModelPath or os.path.join(args.project_path, "merge_outputs", "layer_4_point_cloud.ply")
    model_path = os.path.join(args.project_path, "merge_outputs", "layer_1_point_cloud.ply")
    cams_info = get_camera_info(ImageSimulation_info['Images'])
    
    render_images(model_path, output_path, cams_info, pipeline.extract(args))