'''
    20241212  将图片进行分组,并把路径记录成txt文件
    每个分组图片 不超过500张(默认值)
    txt中储存txt的名字    image_group1.txt,image_group2.txt, ..., image_group3.txt
'''
import os
import sys
from argparse import ArgumentParser

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')   # 图片扩展名

def prepare_out_path(block_path):
    normal_path = os.path.join(block_path, "normals")
    info_path = os.path.join(normal_path, "logs")
    os.makedirs(info_path, exist_ok=True)
    return info_path

def image_group(images_folder, info_folder, max_one_group=500):
    # 初始化分组编号和图片计数器
    group_number = 0
    image_count = 0

    # 用于存储当前组的图片路径
    image_group_txts_list = []
    current_group_images = []

    # 遍历 images_folder 下的所有文件和文件夹
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            # 获取文件的扩展名
            if not file.lower().endswith(image_extensions):
                continue
            current_group_images.append(file)
            image_count += 1

            # 检查当前组的图片数量是否达到500张
            if image_count == max_one_group:
                txt_name = f"image_group{group_number}.txt"
                new_txt = os.path.join(info_folder, txt_name)
                with open(new_txt, 'w') as f:
                    f.write(f"{len(current_group_images)}\n")
                    f.write('\n'.join(current_group_images))
                print(f'Finish image_group{group_number}.txt')
                
                # 重置计数器和图片列表
                group_number += 1
                image_count = 0
                current_group_images = []
                image_group_txts_list.append(txt_name)
    
    # 不足数量的图片组
    if len(current_group_images):
        txt_name = f"image_group{group_number}.txt"
        new_txt = os.path.join(info_folder, txt_name)
        with open(new_txt, 'w') as f:
            f.write(f"{len(current_group_images)}\n")
            f.write('\n'.join(current_group_images))
        print(f'Finish image_group{group_number}.txt')
        group_number += 1
        image_count = 0
        current_group_images = []
        image_group_txts_list.append(txt_name)
    
    # 将所有生成的 txt 记录下来
    task_name = "generate_normal_task.txt"
    task_txt = os.path.join(info_folder, task_name)
    with open(task_txt, 'w') as f:
        f.write(f"{len(image_group_txts_list)}\n")
        f.write('\n'.join(image_group_txts_list))
        print(f'Finish normal task in {task_txt}')
    return image_group_txts_list




if __name__ == '__main__':
    parser = ArgumentParser(description="Image grouping parameters")
    parser.add_argument('--block_path', default="")
    parser.add_argument('--max_one_group', type=int, default=300)
    args = parser.parse_args(sys.argv[1:])

    # 0. 处理文件路径
    image_folder = os.path.join(args.block_path, "images")
    info_folder = prepare_out_path(args.block_path)

    # 1.将默认图片文件夹下的图片进行分组, 逐个保存成txt并给出每个任务txt文件      
    image_group_txts_list = image_group(image_folder, info_folder=info_folder, max_one_group=args.max_one_group)

    print("Image group finish")


    