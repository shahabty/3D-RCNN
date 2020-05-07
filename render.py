#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch3d imports
from pytorch3d.structures import Meshes, Textures
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)
from pytorch3d.renderer.cameras import (
    SfMPerspectiveCameras,
)
from pytorch_3d_add_ons import (
  OpenGLRealPerspectiveCameras,
)

#Misc imports
from utils import str_to_class
import numpy as np
import matplotlib.pyplot as plt
import os

def get_renderer(name,device):
  class_ = str_to_class(name,'render')
  instance = class_(device) if class_ is not None else None
  return instance

class Renderer():
  def __init__(self,device):
    self.device = device
    R, T = look_at_view_transform(1500, 30, 60, device=device)
    K = torch.tensor([[721.5377,0.,609.5593],[0.,721.5377,172.854],[0.,0.,1.]]).to(device)
    
    half_H = 512/2
    half_W = 1382/2
    px_ndc = (K[0,2] - half_W) / half_W
    py_ndc = (K[1,2] - half_H) / half_H
    fx_ndc = K[0,0] / half_W
    fy_ndc = K[1,1] / half_H
    cameras = SfMPerspectiveCameras(focal_length = ((fx_ndc,fy_ndc),),principal_point = ((px_ndc,py_ndc),),device = device,R= R,T=T)
    #cameras = SfMPerspectiveCameras(focal_length = ((K[0,0],K[1,1]),),principal_point = ((K[0,2],K[1,2]),) ,device=device,R=R,T=T)
    self.cameras = cameras
    #Silhouette Renderer
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(image_size=256, blur_radius= 0.0, faces_per_pixel=100,bin_size = 0)
    self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=SoftSilhouetteShader(blend_params=blend_params))

  def render(self,mesh,K = None,distance = 1000*721,elevation = 20.0,azimuth = 80.0):
    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
    # Render the mesh providing the values of R and T.
    silhouete = self.silhouette_renderer(meshes_world=mesh) #R= R,T = T
    return silhouete

  def visualize(self,silhouete,save_dir = None,idx = None):
    silhouete = silhouete.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(silhouete.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    if save_dir is not None:
      plt.savefig(os.path.join(save_dir,str(idx) + '.png'))
