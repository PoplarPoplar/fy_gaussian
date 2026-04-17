import torch
import sys
from torch import nn
import numpy as np
from typing import NamedTuple

from render_utils.graphics_utils import getOrthographicProjectionMatrix, getProjectionMatrix, getWorld2View2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx:float
    cy:float
    width: int
    height: int

class Camera(nn.Module):
    def __init__(self, R, T, width, height, FoVx = 90, FoVy = 90,
                 bottom=1.0, top=1.0, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", is_orthographic=False):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.is_orthographic = is_orthographic

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.image_height = height
        self.image_width = width
        
        self.FoVx = FoVx
        self.FoVy = FoVy
        
        if self.is_orthographic:
            rate = self.image_width / self.image_height
            self.bottom = bottom
            self.top = top
            self.left = self.bottom * rate
            self.right = self.top * rate
            
            self.projection_matrix = getOrthographicProjectionMatrix(self.left, self.right, self.bottom, self.top, self.znear, self.zfar).to(self.data_device)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, primx=0.5, primy=0.5).to(self.data_device)

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]