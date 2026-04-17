import os
import sys
import pythoncom
import win32com.client
import argparse
import xml.etree.ElementTree as ET
import random
import string
import shutil
import math
import json

def create_edge_shortcut(target_url, shortcut_name, shortcut_path):
    # 获取Shell对象
    shell = win32com.client.Dispatch("WScript.Shell")
    
    # 创建快捷方式
    shortcut = shell.CreateShortCut(shortcut_path)
    
    # 设置快捷方式的目标和属性
    shortcut.TargetPath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
    shortcut.Arguments = target_url
    shortcut.IconLocation = shortcut.TargetPath
    shortcut.WorkingDirectory = os.path.dirname(shortcut.TargetPath)
    
    # 保存快捷方式
    shortcut.Save()

def copy_lod_data(source_dir, target_dir, rename=None):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    # 把source_dir copy 到 target_dir下面，同时把source_dir重命名为rename
    shutil.copytree(source_dir, os.path.join(target_dir, rename))

def degrees_to_radians(degrees):
    return degrees * (math.pi / 180)
# def get_project_name(args):
#     #找到args.block_path下面的后缀为".blo"的文件
#     blo_files = [f for f in os.listdir(args.block_path) if f.endswith('.blo')]
#     if len(blo_files) != 1: 
#         raise ValueError("There should be exactly one .blo file in the block_path directory.")
   

#     # 假设XML数据存储在'data.xml'文件中
#     tree = ET.parse(f"{args.block_path}/{blo_files[0]}")
#     root = tree.getroot()

#     # 获取BlockName和ProjectName字段
#     block_name = root.find('BlockName').text
#     project_name = root.find('ProjectName').text

#     print(f'BlockName: {block_name}')
#     print(f'ProjectName: {project_name}')

#     # 再随机生成4位的字符串，作为后缀
#     suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

#     result_name = f"{project_name}_{block_name}_{suffix}"
#     return result_name

def get_project_name(args):
    #把args.block_path路径的最后两个文件夹名拼接起来
    block_path = os.path.normpath(args.block_path)
    block_path = block_path.split(os.sep)
    block_name = block_path[-2] + '_' + block_path[-1]
    # 再随机生成4位的字符串，作为后缀
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    result_name = f"{block_name}_{suffix}"
    return result_name
   

def read_lat_lon(args):
    # 读入经纬度
    # 假设文件名为'gps_info.json'
    file_path = f"{args.block_path}/LOD_outputs/gps_info.json"

    with open(file_path, 'r') as file: 
        data = json.load(file)
        # 提取经纬度信息
        longitude = data['longitude']
        latitude = data['latitude']
        altitude = data['altitude']

    # 将经纬度转换为弧度
    lon_radius = degrees_to_radians(longitude)
    lat_radius = degrees_to_radians(latitude)
    
    # 打印提取的值
    print(f'Altitude: {altitude}')
    print(f'Longitude (radius): {lon_radius}')
    print(f'Latitude (radius): {lat_radius}')

    return altitude, lon_radius, lat_radius
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', required=True, default="")
    parser.add_argument('--web_data_path', required=True, default="")
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])

    # 读入项目名称
    result_name = get_project_name(args)
    copy_lod_data(f"{args.block_path}/LOD_outputs", f"{args.web_data_path}", result_name)

    altitude, lon_radius, lat_radius = read_lat_lon(args)
    url = f"http://localhost/index.html?url=http://localhost/Data/{result_name}/tileset.json&lat={lat_radius}&lng={lon_radius}&alt={altitude}"
    name = "Google Edge Shortcut"
    shortcut_file = os.path.join(args.block_path, f"{name}.lnk")

    create_edge_shortcut(url, name, shortcut_file)
    print(f"快捷方式已创建: {shortcut_file}")
