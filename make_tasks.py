import sys
from argparse import ArgumentParser
from make_gs_chunks import read_valid_chunks, save_valid_chunks, get_tile_bbox
import math
import json
import os

#金字塔训练，由粗到精，是我们的基本思路
#但是由于不能引入太多的图像，也不能使用太小的图像训练
#所的金字塔的层数不是无限的，而是有限的，目前是固定为4层
#如果LOD需要得到更多层级
#那么上面的层级，块不再合并，图像也不再缩小，只是减少点的数量
#下面是示意图：
#    | 1/8 |
#    | 1/8 |
#    / 1/8  \
#   /  1/4   \ 
#  /   1/2    \
# /    1/1     \
#汇总一个下面的vaild_chunks-{chunkname}.txt到一个总的vaild_chunks.txt
def merge_chunks(args):

    last_layer_path = f"{args.block_path}/chunks/layer_{args.layer-1}"
    this_layer_path = f"{args.block_path}/chunks/layer_{args.layer}"
    _, last_vaild_chunks = read_valid_chunks(f"{last_layer_path}/vaild_chunks.txt")

    chunks=[]
    for chunk_name in last_vaild_chunks:
        _, this_vaild_chunks = read_valid_chunks(f"{this_layer_path}/vaild_chunks-{chunk_name}.txt")
        chunks+=this_vaild_chunks

    save_valid_chunks(f"{this_layer_path}/vaild_chunks.txt", chunks)

def load_layer_chunks(args):
    this_layer_path = f"{args.block_path}/chunks/layer_{args.layer}"
    _, this_vaild_chunks = read_valid_chunks(f"{this_layer_path}/vaild_chunks.txt")
    return this_vaild_chunks

def compute_chunk_area(chunk_name, args):
    this_layer_path = f"{args.block_path}/chunks/layer_{args.layer}/"
    chunk = get_tile_bbox(f"{this_layer_path}/{chunk_name}", chunk_name)
    chunk_lens = chunk[1][1] - chunk[1][0]
    area = chunk_lens[0]*chunk_lens[1]
    return area
    
def compute_chunks_point_num(chunks, args, layer_point_area_ratio):
    "根据这一层的总面积和当前块的面积，计算当前块的点数"
    chunk_point_num = {}
    for chunk_name in chunks:
        chunk_point_num[chunk_name] = int(layer_point_area_ratio * compute_chunk_area(chunk_name, args))

    return chunk_point_num


if __name__ == '__main__':
    parser = ArgumentParser(description="Merge chunk list script parameters")
    parser.add_argument('--block_path', required=True, default="")
    parser.add_argument('--layer', required=True, type=int)
   
    args = parser.parse_args(sys.argv[1:])
    
    this_layer_path = f"{args.block_path}/chunks/layer_{args.layer}"
    os.makedirs(this_layer_path, exist_ok=True)
    with open(f"{args.block_path}/block_gs_info.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    if args.layer >= data['layer_num']:
        print("Layer out of range")
        with open(f"{this_layer_path}/task_list.txt", 'w') as file:
            file.write("0")
        sys.exit(0)

    # 找到data['layer_infos']中image_scale为当前layer的项
    layer_info = next((info for info in data['layer_infos'] if info['layer'] == args.layer), None)
    layer_point_area_ratio = layer_info['point_area_ratio']
    layer_image_scale = layer_info['image_scale']
    layer_image_resolution = layer_info['image_resolution']

    #合并或加载所有要做任务块
    if args.layer != 0:
        merge_chunks(args)
    chunk_names = load_layer_chunks(args)
    #根据面积比例计算每个块的点数
    chunk_point_num = compute_chunks_point_num(chunk_names,args, layer_point_area_ratio)

    tasks = []


    for chunk_name, point_num in chunk_point_num.items():
        #因为正常的金字塔层层级是用图像的分辨率来控训练的分辨率的，所以scale都是1
        image_folder = f"images_{layer_image_scale}" if layer_image_scale > 1 else "images"
        tasks.append(f"layer_{args.layer} {chunk_name} {point_num} {image_folder} {layer_image_resolution}")
    with open(f"{this_layer_path}/task_list.txt", 'w') as file:
        file.write(f"{len(tasks)}\n")
        file.write('\n'.join(tasks))