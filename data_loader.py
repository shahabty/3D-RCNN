#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#Misc imports
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import cv2

#Pytorch3d imports
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.transforms import Rotate, Translate

#Local imports
from utils import str_to_class


def get_dataset(name,mode,data_path,device):
  mode = 'train' if mode == 'val' else 'train'
  class_ = str_to_class(name,'data_loader')
  instance = class_(mode,data_path,device)
  return instance or None


class KITTI(Dataset):
  def __init__(self,mode,data_path,device):
    self.mode = mode
    self.data_path = data_path
    self.device = device
    self.data = dict()
        
    image_path = os.path.join(data_path,'data_object_image_2',mode + 'ing','image_2/')
    image_samples = os.listdir(image_path)
    image_data = [os.path.join(data_path,'data_object_image_2',mode + 'ing','image_2/') + s for s in sorted(image_samples)]
    self.data['image'] = image_data

    calib_path = os.path.join(data_path,'data_object_calib',mode + 'ing','calib/')
    calib_samples = os.listdir(calib_path)
    calib_data = [os.path.join(data_path,'data_object_calib',mode + 'ing','calib/') + s for s in sorted(calib_samples)]
    self.data['calib'] = calib_data
    
    if mode == 'train':
      label_path = os.path.join(data_path,'data_object_label_2','training','label_2/')
      label_samples = os.listdir(label_path)
      label_data = [os.path.join(data_path,'data_object_label_2','training','label_2/') + s for s in sorted(label_samples)]
      self.data['label'] = label_data

      depth_path = os.path.join(data_path,'depth_2_multiscale/')
      depth_samples = os.listdir(depth_path)
      depth_data = [os.path.join(data_path,'depth_2_multiscale/') + s for s in sorted(depth_samples)]
      self.data['depth'] = depth_data
    
    

  def __getitem__(self,idx):
    input_data = {'image':None,'depth':None,'calib':None,'label':None}
    image_path = self.data['image'][idx]
    calib_path = self.data['calib'][idx]
    input_data['image'] = torch.from_numpy(np.array(cv2.imread(image_path)))#.permute(2,0,1)
    input_data['calib'] = self.load_calib(calib_path)

    if self.mode == 'train':
      label_path = self.data['label'][idx]
      depth_path = self.data['depth'][idx]      
      input_data['label'] =  self.load_label(label_path)
      #the depth is pre-processed, so we just load it from png format
      input_data['depth'] = torch.from_numpy(np.array(cv2.imread(depth_path)))#.permute(2,0,1)
    return input_data

  def load_calib(self,calib_path):
    return None

  def load_label(self,label_path):
    return None

  def __len__(self):
    return len(self.data['image'])


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

