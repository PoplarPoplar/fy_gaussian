import os
import struct
import argparse
from pathlib import Path
import sys
import math

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_image_paths_from_bin(bin_file):
    """Read image paths from a binary file."""
    image_paths = []
    try:
        with open(bin_file, 'rb') as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                # Read the basic properties
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi"
                )
                image_id = binary_image_properties[0]

                # Read the image name, which is null-terminated
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":  # ASCII null character
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]

                image_paths.append(image_name)

                # Skip the 2D points data
                num_points2D = read_next_bytes(fid, 8, "Q")[0]
                fid.read(24 * num_points2D)  # Skip points data
    except FileNotFoundError:
        print(f"Error: The binary file '{bin_file}' was not found.")
        raise
    except struct.error as e:
        print(f"Error reading binary file: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return image_paths

def divide_array(array, task_num, min_elements_per_task=400):
    total_length = len(array)
    
    # 计算基本的任务数量
    num_tasks = total_length // min_elements_per_task
    
    # 如果基本的任务数量大于或等于task_num，则使用task_num
    if num_tasks >= task_num:
        num_tasks = task_num
    else:
        # 否则，向上取整得到任务数量
        num_tasks = math.ceil(total_length / min_elements_per_task)
    
    # 分割数组
    tasks = [array[i * (total_length // num_tasks):(i + 1) * (total_length // num_tasks)] 
             for i in range(num_tasks)]
    
    return tasks
def save_image_paths_to_files(image_paths, prefix, num_task, output_dir):
    """Split image paths into text files with given prefix and save them."""
   
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Add prefix to image paths
    absolute_image_paths = [os.path.join(prefix, path) for path in image_paths]
    
    # Calculate the size of each split    
    i = 0 
    task_list = []
    for chunk in divide_array(image_paths, num_task):
        file_path = os.path.join(output_dir, f"image_paths_part_{i+1}.txt")
        task_list.append(f"image_paths_part_{i+1}.txt")
        i+=1
        with open(file_path, 'w') as file:
            for image_name in chunk:
                file.write(f"{os.path.join(prefix, image_name)}\n")
            
    task_file_path = os.path.join(output_dir, "depth-infer-task.txt")
    with open(task_file_path, 'w') as master_file:
        # 写入列表中元素的数量
        master_file.write(f"{len(task_list)}\n")
        # 写入列表中的每个文件名
        for file_name in task_list:
            master_file.write(f"{file_name}\n")
  

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Read and split image paths from a binary file.")
    parser.add_argument('--image-bin-file', type=str, required=True, help="Path to the binary file containing image paths.")
    parser.add_argument('--image-dir', type=str, required=True, help="Prefix directory to add to each image path.")
    parser.add_argument('--num-task', type=int, required=True, help="Number of text files to split paths into.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save the split text files.")
    args = parser.parse_args()

    try:
        # Read image paths from the binary file
        image_paths = read_image_paths_from_bin(args.image_bin_file)
        print(f"Found {len(image_paths)} images in the binary file.")

        # Split paths into specified number of files
        save_image_paths_to_files(image_paths, args.image_dir, args.num_task, args.output_dir)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
