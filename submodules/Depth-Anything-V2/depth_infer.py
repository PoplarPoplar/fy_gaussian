import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

def get_relative_output_path(input_path, root_dir, outdir):
    """
    Create a mirrored output path based on the input path.
    
    :param image_list_txt: Full path of the input image.
    :param image_dir: Root directory of the input images.
    :param outdir: Base directory for output images.
    :return: Full path where the output image should be saved.
    """
    relative_path = os.path.relpath(input_path, start=root_dir)
    output_path = os.path.join(outdir, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--image-list-txt', type=str,required=True, help='Path to a single image or text file containing image paths.')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the model.')
    parser.add_argument('--outdir', type=str, required=True,help='Directory to save output depth images.')
    parser.add_argument('--image-dir', type=str, required=True, help='Root directory of the images to maintain directory structure.')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory where the model .pth file is stored.')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Hardcoded model configuration for "vitl"
    model_config = {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    }
    
    depth_anything = DepthAnythingV2(**model_config)
    
    # Load the model .pth file from the specified model directory
    model_path = os.path.join(args.model_dir, 'depth_anything_v2_vitl.pth')
    depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.image_list_txt):
        if args.image_list_txt.endswith('txt'):
            with open(args.image_list_txt, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.image_list_txt]
    else:
        filenames = glob.glob(os.path.join(args.image_list_txt, '**/*'), recursive=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Hardcoded behavior for grayscale and pred-only
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        # Calculate the output path for the depth map, maintaining directory structure
        output_path = get_relative_output_path(filename, args.image_dir, args.outdir)
        output_file = os.path.splitext(output_path)[0] + '.png'
        
        cv2.imwrite(output_file, depth)
