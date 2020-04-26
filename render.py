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
    #Perspective camera
    R, T = look_at_view_transform(700, 20.0, 30.0) 
    cameras = OpenGLPerspectiveCameras(device=device,R=R,T=T)

    #Silhouette Renderer
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(image_size=256, blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, faces_per_pixel=100,)
    self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=SoftSilhouetteShader(blend_params=blend_params))

    #Phong Renderer
    raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1,)
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    self.phong_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=HardPhongShader(device=device, lights=lights))

  def render(self,mesh,distance = 800,elevation = 20.0,azimuth = 10.0):

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)

    # Render the mesh providing the values of R and T. 
    silhouete = self.silhouette_renderer(meshes_world=mesh,R=R,T=T)
    image_ref = self.phong_renderer(meshes_world=mesh,R=R,T=T)
    return silhouete, image_ref
    
  def visualize(self,silhouete,image_ref,save_dir = None,idx = None):
    silhouete = silhouete.cpu().numpy()
    image_ref = image_ref.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouete.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)
    if save_dir is not None:
      plt.savefig(os.path.join(save_dir,str(idx) + '.png'))
