import json
import os
from math import radians
from datetime import datetime
from utils.aabb_utils import getAABB

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def json_write(tileset_json, save_path):
    json_path = os.path.join(save_path, "tileset.json")
    with open(json_path, 'w') as f:
        json.dump(tileset_json, f, indent=4)

def create_child_jsoninfo(output_path, lodtree, level, geo_list):
    """
    output_path: str, 成果输出路径
    lodtree: class create_LOD, 单个 tile 的 LODtree
    level: int, 单个 tile 的总层数
    geo_list: list, 预计算的几何误差列表
    """
    path = os.path.relpath(lodtree.save_path, output_path)
    # path = os.path.normpath(path)
    path = path.replace("\\", "/")
    aabb = getAABB(lodtree.min_point, lodtree.max_point)
    
    if lodtree.level < level - 1:
        tile_json = {
            "boundingVolume":{
                "box": aabb.get_boundingVolume_box()
            },
            "children":[],
            "content": {
                "uri": f"./{path}/{lodtree.name}.bin"
            },
            "geometricError": geo_list[-1],
            "refine": "REPLACE"
        }
        for child_key, child_node in lodtree.child.items():
            child_tile_json = create_child_jsoninfo(output_path, child_node, level, geo_list[:-1])
            tile_json['children'].append(child_tile_json)
    else:
        tile_json = {
            "boundingVolume":{
                "box": aabb.get_boundingVolume_box()
            },
            "content": {
                "uri": f"./{path}/{lodtree.name}.bin"
            },
            "geometricError": geo_list[-1]
        }
    return tile_json

def create_tile_jsoninfo(output_path, lodtree, level, geo_list):
    """
    得到单个 lod_tile 的 json
    output_path: str, 成果输出路径
    lodtree: class create_LOD, 单个 tile 的 LODtree
    level: int, 单个 tile 的总层数
    geo_list: list, 算好的各层级几何误差
    """
    path = os.path.relpath(lodtree.save_path, output_path)
    path = path.replace("\\", "/")
    # path = os.path.normpath(path)
    aabb = getAABB(lodtree.min_point, lodtree.max_point)
    tile_json = {
        "boundingVolume":{
            "box": aabb.get_boundingVolume_box()
        },
        "children":[],
        "content": {
            "uri": f"./{path}/{lodtree.name}.bin"
        },
        "geometricError": geo_list[-1],
        "refine": "REPLACE"
    }
    
    for child_key, child_node in lodtree.child.items():
        child_tile_json = create_child_jsoninfo(output_path, child_node, level, geo_list[:-1])
        tile_json['children'].append(child_tile_json)
    return tile_json

def create_project_json(project_name, output_path, transform_matrix, min_point, max_point, tile_json_list, geo_list):
    """
    得到整个场景的 json
    project_name: str, 项目名称
    output_path: str, json 文件输出路径
    transform_matrix: np.array(4,4)
    min_point, max_point: np.array(3), AABB 的最小点与最大点
    tile_json_list: list, 各个 lodtile 的 json 信息
    geo_list: list, 计算获得的几何误差列表
    """
    current_date = datetime.now()
    formatted_date = current_date.strftime("%m/%d/%Y")
    aabb = getAABB(min_point, max_point)
    tileset_json = {
        "asset": {
            "extras": {
                "Dataset": project_name,
                "Date": formatted_date
            },
            "gltfUpAxis": "Z",
            "version": "1.0"
        },
        "root": {
            "boundingVolume": {
                "box": aabb.get_boundingVolume_box()
            },
            "children": [],
            "geometricError": geo_list[-1],
            "transform": [
                    transform_matrix[0][0],
                    transform_matrix[1][0],
                    transform_matrix[2][0],
                    transform_matrix[3][0],
                    transform_matrix[0][1],
                    transform_matrix[1][1],
                    transform_matrix[2][1],
                    transform_matrix[3][1],
                    transform_matrix[0][2],
                    transform_matrix[1][2],
                    transform_matrix[2][2],
                    transform_matrix[3][2],
                    transform_matrix[0][3],
                    transform_matrix[1][3],
                    transform_matrix[2][3],
                    transform_matrix[3][3],
                ]
        },
    }
    
    for i in range(len(tile_json_list)):
        tileset_json["root"]['children'].append(tile_json_list[i])
    
    json_path = os.path.join(output_path, "tileset.json")
    with open(json_path, 'w') as f:
        json.dump(tileset_json, f, indent=4)

def create_tile_json(output_path, lodtree, level, method, error_scale):
    """
    为单个 tile 创建并保持 json 文件信息
    output_path: str, 成果输出路径
    lodtree: class create_LOD, 单个 tile 的 LODtree
    level: int, 单个 tile 的总层数
    method: int, 几何误差计算方法
    error_scale: float, 几何误差缩放比例
    """
    # path = os.path.normpath(path)
    aabb = getAABB(lodtree.min_point, lodtree.max_point)
    if method == 0:
        error = aabb.get_radius()
    elif method == 1:
        error = aabb.get_radius_withoutz()
    tileset_json = {
        "asset": {
            "gltfUpAxis": "Z",
            "version": "1.0"
        },
        "root":{
            "boundingVolume":{
                "box": aabb.get_boundingVolume_box()
            },
            "children":[],
            "content": {
                "uri": f"./{lodtree.name}.bin"
            },
            "geometricError": error  / error_scale,
            "refine": "REPLACE"
        }     
    }
    path = os.path.relpath(lodtree.save_path, output_path)
    path = path.replace("\\", "/")
    tile_json = {
        "boundingVolume":{
            "box": aabb.get_boundingVolume_box()
        },
        "children":[],
        "content": {
            "uri": f"./{path}/tileset.json"
        },
        "geometricError": error  / error_scale
    }
    
    for child_key, child_node in lodtree.child.items():
        child_tile_json = create_child_jsoninfo(lodtree.save_path, child_node, level, method, error_scale)
        tileset_json['root']['children'].append(child_tile_json)
    
    json_path = os.path.join(lodtree.save_path, "tileset.json")
    with open(json_path, 'w') as f:
        json.dump(tileset_json, f, indent=4)
        
    return tile_json

def create_gps_info(output_path, lon, lat, altitude):
    """
    生成含 gps 信息的 json 文件
    output_path: str, 成果输出路径
    lon, lat, altitude: float
    """
    tileset_json = {
        "longitude": lon,
        "latitude": lat,
        "altitude": altitude,
        "lon(rad)": radians(lon),
        "lat(rad)": radians(lat)
    }
    
    json_path = os.path.join(output_path, "gps_info.json")
    with open(json_path, 'w') as f:
        json.dump(tileset_json, f, indent=4)
    
    print(f"gps info is written to {output_path}/gps_info.json ")