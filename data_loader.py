import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

#my imports
import importlib
import logging
import os

def get_dataset(name,mode,data_path,device):
  try:
    module = importlib.import_module('data_loader')
    try:
      class_ = getattr(module, name)
    except AttributeError:
      logging.error('The dataset is not implemented!')
  except ImportError:
    logging.error('Module does not exist!')
  instance = class_(mode,data_path,device)
  return instance or None


class Cars(nn.Module):
  def __init__(self,mode,data_path,device):
    self.data_path = data_path
    self.device = device
    self.data = []
    for c in os.listdir(data_path):
      for f in os.listdir(os.path.join(data_path,c)):
        if f.endswith(".obj"):
          self.data.append(os.path.join(data_path,c,f)) 
           
  def __getitem__(self,x):
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(self.dataset[x])
    faces = faces_idx.verts_idx
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(self.device))

    mesh_obj= Meshes(verts=[verts.to(self.device)],faces=[faces.to(self.device)],textures=textures)
    return mesh_obj 
  def __len__(self):
    return len(self.data)
