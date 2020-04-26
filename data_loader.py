#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#Misc imports
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import logging
import os
import numpy as np


#Pytorch3d imports
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.transforms import Rotate, Translate

#Local imports
from utils import str_to_class


def get_dataset(name,mode,data_path,device):
  class_ = str_to_class(name,'data_loader')
  instance = class_(mode,data_path,device)
  return instance or None


class KITTI(Dataset):
  def __init__(self,mode,data_path,device):
    self.mode = mode
    self.data_path = data_path
    self.device = device 
    self.image_data = []
    self.label_data = []
    self.calib_data = []
    
    image_path = os.path.join(data_path,data_object_image_2,mode + 'ing','image_2')
    image_samples = os.listdir(image_path)
    calib_path = os.path.join(data_path,data_object_calib,mode + 'ing',c'calib')
    calib_samples = os.listdir(calib_path)
    if mode == 'train':
      label_path = os.path.join(data_path,data_object_label_2,'training','label_2')
      label_samples = os.listdir(label_path)

    for image, calib in zip(): 
      if image_samples.endswith('.png'):
        


  def __getitem__(self,x):


  def __len__(self):



class Cars_3D(Dataset):
  def __init__(self,mode,data_path,device):
    self.mode = mode
    self.data_path = data_path
    self.device = device
    self.data = []
    for c in os.listdir(data_path):
      for f in os.listdir(os.path.join(data_path,c)):
        if f.endswith(".obj"):
          self.data.append(os.path.join(data_path,c,f)) 
           
  def __getitem__(self,x):
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(self.data[x])
    faces = faces_idx.verts_idx
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(self.device))
    mesh_obj= Meshes(verts=[verts.to(self.device)],faces=[faces.to(self.device)],textures=textures)
    return mesh_obj
 
  def __len__(self):
    return len(self.data)
