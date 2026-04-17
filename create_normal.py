import struct
import os
import torch
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
from collections import defaultdict
import numpy as np
from PIL import Image


def direct_collate(x):
    return x

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def store_render_results(dataset, opt, pipe, load_ply_path):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(load_ply_path)

    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 4, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
    all_normals_dir = os.path.join(dataset.model_path, "all_normals")
    os.makedirs(all_normals_dir, exist_ok=True)

    tt = time.time()
    with torch.no_grad():
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                uid = viewpoint_cam.uid
                background = torch.rand((3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                # 此视角上渲染的法向图   注意变成反向 与stable normal 进行对应
                normal = - render(viewpoint_cam, gaussians, pipe, background)["normal"]   # [-1, 1]
                normal_arr = normal.detach().cpu().numpy()
                normal_arr = (normal_arr.clip(-1, 1) + 1)/ 2
                normal_arr = (normal_arr * 255).astype(np.uint8)
                normal_arr = normal_arr.transpose(1, 2, 0)
                
                pred_normal = Image.fromarray(normal_arr)
                pred_normal.save(os.path.join(all_normals_dir, viewpoint_cam.image_name))
                print(f"{viewpoint_cam.image_name} Done")


    print(f"Final time is {time.time() - tt :.4f}")
            
    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_ply_path", type=str, default="")
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    store_render_results(lp.extract(args), op.extract(args), pp.extract(args), load_ply_path=args.load_ply_path)

    # All done
    print("\nRender results complete.")
