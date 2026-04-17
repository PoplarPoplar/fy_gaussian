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


# ex
# python create_vpc.py -s D:\\block_gs_test\new_test1021 -i D:\block_gs_test\new_test1021\images -m D:\block_gs_test\new_test1021\outputs_vpc_1030 --load_ply_path D:\block_gs_test\new_test1021\outputs1024\point_cloud\iteration_final\point_cloud.ply

class PointCloud:
    def __init__(self, points=[], point_views=[]):
        self.points = points  # List to store points
        self.point_views = point_views  # List to store views for each point

    def IsEmpty(self):
        return len(self.points) == 0

    def SaveVpc(self, file_name):
        '''
           write vpc file 仿照AI3D vpc文件格式
        '''
        if self.IsEmpty():
            return False

        with open(file_name, 'wb') as ofs:
            # Write the number of points
            n_points = len(self.points)
            ofs.write(struct.pack('I', n_points))

            for i in range(n_points):
                point = self.points[i]
                # Uncomment the following lines if you need to apply relative translation
                # point.x += relative_trans[0]
                # point.y += relative_trans[1]
                # point.z += relative_trans[2]

                # Write the coordinates of the point
                ofs.write(struct.pack('f', point[0]))
                ofs.write(struct.pack('f', point[1]))
                ofs.write(struct.pack('f', point[2]))

                views = self.point_views[i]
                n_views = len(views)
                # Write the number of views for the point
                ofs.write(struct.pack('I', n_views))

                # Write the view IDs
                for view_id in views:
                    ofs.write(struct.pack('I', view_id))

        return True
    
    def LoadVpc(self, file_name: str) -> bool:
        '''
           read vpc file
        '''
        if not file_name.endswith('vpc'):
            return False
        
        # 对points以及point_views重新进行赋值
        self.points = []
        self.point_views = []

        try:
            with open(file_name, 'rb') as ifs:
                # Read the number of points
                n_points_data = ifs.read(struct.calcsize('I'))
                if len(n_points_data) != struct.calcsize('I'):
                    return False
                n_points = struct.unpack('I', n_points_data)[0]
                if n_points < 1:
                    return False

                for _ in range(n_points):
                    # Read the coordinates of the point
                    x_data = ifs.read(struct.calcsize('f'))
                    y_data = ifs.read(struct.calcsize('f'))
                    z_data = ifs.read(struct.calcsize('f'))
                    if len(x_data) != struct.calcsize('f') or \
                       len(y_data) != struct.calcsize('f') or \
                       len(z_data) != struct.calcsize('f'):
                        return False
                    x, y, z = struct.unpack('fff', x_data + y_data + z_data)
                    self.points.append([x, y, z])

                    # Read the number of views for the point
                    n_views_data = ifs.read(struct.calcsize('I'))
                    if len(n_views_data) != struct.calcsize('I'):
                        return False
                    n_views = struct.unpack('I', n_views_data)[0]

                    views = []
                    for _ in range(n_views):
                        # Read the view IDs
                        view_id_data = ifs.read(struct.calcsize('I'))
                        if len(view_id_data) != struct.calcsize('I'):
                            return False
                        view_id = struct.unpack('I', view_id_data)[0]
                        views.append(view_id)
                    self.point_views.append(views)

        except IOError:
            return False

        return True

def saveCamera2Ply(path, vertices):
    from plyfile import PlyData, PlyElement
    vertices = np.array(vertices)

    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]

    elements = np.empty(vertices.shape[0], dtype=dtype_full)
    attributes = np.concatenate((vertices, ), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def readply(ply_path):
    from plyfile import PlyData, PlyElement

    # 读取PLY文件
    plydata = PlyData.read(ply_path)

    # 获取顶点元素
    vertices = plydata['vertex']

    # 打印顶点数量
    print(f"Number of vertices: {len(vertices)}")

    # 遍历并打印每个顶点的属性
    for i, vertex in enumerate(vertices):
        print(f"{i}: Vertex: x = {vertex[0]}, y = {vertex[1]}, z = {vertex[2]}")

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

    indices = None  
    # data record 
    cam_pos = [None] * scene.num_cameras
    point_views = defaultdict(list)   # Gauss点看到的camera_id
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 4, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

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

                # 20241101 注意这里如果用["visibility_filter"]会在AI3D构网过程的问题
                visibility_filter = render(viewpoint_cam, gaussians, pipe, background)["maxweight_indices"].flatten().unique().tolist()

                for point_id in visibility_filter:
                    point_views[point_id].append(uid)
                cam_pos[uid] = viewpoint_cam.camera_center.tolist()
                print(f"{uid} {viewpoint_cam.image_name} Done {cam_pos[uid]}")

        # Cameras
        saveCamera2Ply(path=os.path.join(dataset.model_path, "camera.ply"), vertices=cam_pos)

        # Points
        points = gaussians.get_xyz.tolist()
        points_class = PointCloud(points=points, point_views=point_views)
        # check
        # for i in range(len(points)):
        #     print(f"{i}: {points[i]} {len(point_views[i])}")

        points_class.SaveVpc(os.path.join(dataset.model_path, "test.vpc"))
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
