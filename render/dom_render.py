# 2024.11.13
# 利用 3DGS 模型生成正射影像
# 
# 使用方法
# dom_render.py -p project_path
# 
# 若不设置 -g 参数，高斯模型会读取 merge_outputs 下最精细模型所对应的 ply 文件，若需要设置 -g 参数，指定到ply文件
# 可以通过 -o 参数指定输出路径，默认会在 project_path 文件夹下创建文件夹 TDOM_result 保存成果
# 可以通过 -r 参数指定正射影像分辨率，默认会读取 at.xml 文件中的 gsd
# 可通过 --frame 参数设置分幅输出， --pixel 参数用于指定分幅输出每张图片的分辨率最大大小，即最大为 pixel * pixel
# 
# 采用 tifffile 库保存 tiff 文件， 因为 pillow 库无法处理较大的 tiff 文件
# 对于 tiffile 采用了 jpeg 压缩（有损）
# 打包时采用命令 pyinstaller --collect-all=imagecodecs dom_render.py ，否则运行会报错 
# imagecodecs.imagecodecs.DelayedImportError: could not import name 'jpeg8_encode' from 'imagecodecs'
# -puzhou -r 0.005 time:5.41844296秒
# -110kv -r 0.005 time:10.40371323秒
# python ./render/dom_render.py -g D:\code\program\blockgs-master\output\test\110kv_16\point_cloud\iteration_final\point_cloud.ply -o D:\code\program\blockgs-master\output\test\DOM --rate 0.05  -p D:\code\program\blockgs-master\output\test\110kv_16
import time
import os
import torch
import math
import torchvision
import shutil
import json
import numpy as np
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from plyfile import PlyData
from PIL import Image
import tifffile
import imagecodecs

from gauss_render import tdom_render
from render_utils.cameras_utils import Camera
from render_utils.Gaussian import GaussianModel
from render_arguments import PipelineParams, ModelParams, get_combined_args

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_tfw(filename, pixel_width, pixel_height, x_origin, y_origin):
    """
    写入TFW文件，坐标为
    :param filename: TFW文件的输出路径
    :param pixel_width: 图像的像素宽度（单位：地图单位/像素）
    :param pixel_height: 图像的像素高度（单位：地图单位/像素）
    :param x_origin: 左上角的X坐标
    :param y_origin: 左上角的Y坐标
    """
    with open(filename, 'w') as f:
        f.write(f"{pixel_width}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{-pixel_height}\n")
        f.write(f"{x_origin}\n")
        f.write(f"{y_origin}\n")

def calculate_AABB_from_points(gs_ply):
    """
    根据高斯椭球坐标计算包围盒大小
    # TODO 考虑高斯椭球 scale 的均值
    gs_ply: Plydata['vertex']
    """
    points = np.array([gs_ply['x'], gs_ply['y'], gs_ply['z']]).T
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    return max_point, min_point

def render_block(cam_info: Camera, gaussian: GaussianModel, pipeline : PipelineParams):
    # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg_color = [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rendering = tdom_render(cam_info, gaussian, pipeline, background)["render"]
    return rendering

def render_dom(output_path, width, height, min_point, max_point, rate, normalizer, id, gaussian, pipeline):
    '''
    分幅渲染
    output_path: str, 输出路径
    width: int, 图片宽度
    height: int, 图片高度
    min_point, max_point: np.array, 场景AABB范围
    rate: float, 像素代表的实际距离长度
    normalizer: np.array, 局部坐标系原点对应的 2000 国家大地投影
    id: str, 当前分幅id名称
    gaussian: GaussianModel, 高斯模型
    pipeline: 高斯模型参数
    '''
    with torch.no_grad():
        # step1 ：分块渲染，每个小块为 2048 x 2048
        block_size = 2048
        num_blocks_x = math.ceil(width / block_size)
        num_blocks_y = math.ceil(height / block_size)
        
        temp_path = os.path.join(output_path, "temp_dom")
        os.makedirs(temp_path, exist_ok=True)
        
        block_images = []
        for j in range(num_blocks_y):
            for i in range(num_blocks_x):
                block_min_x = i * block_size
                block_max_x = min((i + 1) * block_size, width)
                block_min_y = j * block_size
                block_max_y = min((j + 1) * block_size, height)

                real_min_y = min_point[1] + block_min_y * rate
                real_max_y = min_point[1] + block_max_y * rate
                real_min_x = max_point[0] - block_max_x * rate
                real_max_x = max_point[0] - block_min_x * rate
                block_length, block_width = real_max_x - real_min_x, real_max_y - real_min_y

                # 获取相机
                Tx, Ty = real_min_x + block_length / 2, real_min_y + block_width / 2
                R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # 相机沿 x 轴翻转 180° 即可实现正对地面
                C_w = np.array([Tx, Ty, max_point[2]+10])  # 相机中心所在坐标
                T = -np.dot(R, C_w)
                camera = Camera(R=R, T=T, bottom=-block_width / 2, top=block_width / 2, width=block_max_x - block_min_x, height=block_max_y - block_min_y, is_orthographic=True)

                # 渲染图片
                block_image = render_block(camera, gaussian, pipeline)
                block_output_path = os.path.join(temp_path, f"block_{num_blocks_y - j - 1}_{num_blocks_x - i - 1}.png")
                torchvision.utils.save_image(block_image, block_output_path)
                block_images.append(block_output_path)

        # step2 ：将所有小块图像合并为大图像并保存图片
        merged_image_data = np.zeros((height, width, 3), dtype=np.uint8)

        current_y = 0  # 当前的累计高度
        for j in range(num_blocks_y):
            current_x = 0  # 每行的起始 x 坐标
            for i in range(num_blocks_x):
                block_path = block_images[(num_blocks_y - j - 1) * num_blocks_x + (num_blocks_x - i - 1)]
                block_image = Image.open(block_path)
                
                block_data = np.array(block_image)
                merged_image_data[current_y:current_y + block_data.shape[0], 
                                current_x:current_x + block_data.shape[1]] = block_data

                # 更新当前行的累计宽度
                current_x += block_data.shape[1]

            # 更新当前列的累计高度
            current_y += block_data.shape[0]

        # step 3 : 输出成果
        
        # 保存为 TIFF 文件
        output_tiff_path = os.path.join(output_path, f"result_{id}.tiff")
        tifffile.imwrite(output_tiff_path, merged_image_data, compression='jpeg')
        print(f"Saved merged image to {output_tiff_path}")
        
        # 输出 TFW
        r = rate
        position_min = normalizer + min_point
        position_max = normalizer + max_point
        output_tfw_path = os.path.join(output_path, f"result_{id}.tfw")
        write_tfw(output_tfw_path, r, r, position_min[0], position_max[1])
        print(f"Saved tfw file to {output_tfw_path}")

def generate_dom(normalizer, model_path, output_path, rate, frame, pixel, pipeline):
    '''
    生成 DOM 影像和 TFW 文件
    normalizer: np.array, 局部坐标系原点对应的 2000 国家大地投影
    model_path: str, 模型路径
    output_path: str, 成果保存路径
    rate: float, 像素代表的实际距离长度
    frame: float, 是否分幅
    pixel: int, 分幅后照片最大像素
    pipeline: 高斯模型参数
    '''
    with torch.no_grad():
        # step1 ：读取点云计算包围盒，加载高斯模型
        gaussian = GaussianModel(0)
        plydata = PlyData.read(model_path)
        gaussian.load_ply(model_path)
        
        max_point, min_point = calculate_AABB_from_points(plydata['vertex'])
        center = (max_point[:2] + min_point[:2]) / 2
        
        # step2 ：计算场景大小以及像素数量
        scene_length, scene_width = max_point[0] - min_point[0],  max_point[1] - min_point[1]
        pixels_width = math.ceil(scene_length / rate)
        # print("pixels_width = ",pixels_width)
        pixels_height = math.ceil(scene_width / rate)
        # print("pixels_heght = ", pixels_width)
        # step3 ：根据计算出来的像素反算包围盒并均匀扩大包围盒，使原包围盒在相片中间
        new_scene = np.array([pixels_width * rate, pixels_height * rate])
        new_max_point = np.array([center[0] + new_scene[0] / 2, center[1] + new_scene[1] / 2, max_point[2]])
        # print("new_max_point = ", new_max_point)
        new_min_point = np.array([center[0] - new_scene[0] / 2, center[1] - new_scene[1] / 2, min_point[2]])
        # print("new_min_point = ", new_min_point)

        # step4 : 根据给定的参数计算分块数量，并分幅分块渲染
        if frame:
            num_blocks_x = math.ceil(pixels_width / pixel)
            num_blocks_y = math.ceil(pixels_height / pixel)
            
            for j in range(num_blocks_y):
                for i in range(num_blocks_x):
                    block_min_x = i * pixel
                    block_max_x = min((i + 1) * pixel, pixels_width)
                    block_min_y = j * pixel
                    block_max_y = min((j + 1) * pixel, pixels_height)
                    
                    real_min_x = new_min_point[0] + block_min_x * rate
                    real_max_x = new_min_point[0] + block_max_x * rate
                    real_min_y = new_min_point[1] + block_min_y * rate
                    real_max_y = new_min_point[1] + block_max_y * rate
                    
                    render_dom(output_path, block_max_x - block_min_x, block_max_y - block_min_y, 
                               np.array([real_min_x, real_min_y, new_min_point[2]]), 
                               np.array([real_max_x, real_max_y, new_max_point[2]]), 
                               rate, normalizer, f"{num_blocks_y - j - 1}_{i}", gaussian, pipeline)
        else:
            render_dom(output_path, pixels_width, pixels_height, new_min_point, new_max_point, rate, normalizer, 0, gaussian, pipeline)
        shutil.rmtree(os.path.join(output_path, "temp_dom"))

if __name__ == "__main__":
    start_time = time.time()
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--dom", type=int, help="1 generate TDOM, 0 not", default=1) # 方便快速版集成设置的参数
    parser.add_argument("-p", "--project_path", type=str, help="the project local in", required=True)
    parser.add_argument("-g", "--gaussian_model_path", type=str, help="the gaussian model local in", default="")
    parser.add_argument("-o", "--output_path", type=str, default="")
    
    parser.add_argument("--rate", type=float, help="distance (m) represented by a pixel")
    parser.add_argument("--frame", action="store_true", help="whether to frame", default=False)
    parser.add_argument("--pixel", type=int, help="framing the pixel size of each photo", default=10000)
    args = parser.parse_args()
    
    if args.dom == 1:
        # 获取局部坐标系原点对应的坐标
        BoundingBox_info_json_path = os.path.join(args.project_path, "BoundingBox.json")
        if not os.path.exists(BoundingBox_info_json_path):
            assert False, f"There doesn't exist BoundingBox.json in {args.project_path}!"
        BoundingBox_info = read_json_file(BoundingBox_info_json_path)
                
        normalizer = np.array([BoundingBox_info["Normalizer"]["origin"]["x"], 
                            BoundingBox_info["Normalizer"]["origin"]["y"],
                            BoundingBox_info["Normalizer"]["origin"]["z"]])
        print("normalizer = ",normalizer)
        
        if args.gaussian_model_path == '':
            # 默认读取最精细一层模型结果
            models_path = os.path.join(args.project_path, "merge_outputs")
            ply_files = sorted(f for f in os.listdir(models_path) if f.endswith('.ply'))
            model_path = os.path.join(models_path, ply_files[-1])
        else:
            model_path = args.gaussian_model_path
        
        # 读取 at.xml 文件中的 分辨率
        if args.rate == None:
            at_xml_path = os.path.join(args.project_path, "at.xml")
            root = ET.parse(at_xml_path).getroot()
            gsd_value = float(root.find('GSD').text)
            rate = math.floor(gsd_value * 100) / 100
        else:
            rate = args.rate

        output_path = os.path.join(args.project_path, "TDOM_result") if args.output_path == '' else args.output_path
        print("Output folder: {}".format(output_path))
        
        generate_dom(normalizer, model_path, output_path, rate, args.frame, args.pixel, pipeline.extract(args))
        end_time = time.time()
        print(f"运行时间：{end_time - start_time:.8f}秒")