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

#Detectron2 imports
import detectron2.data.transforms as T

#Pytorch3d imports
from pytorch3d.io import load_obj,load_objs_as_meshes
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
    self.transform_gen = T.ResizeShortestEdge([800, 800], 1333)
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
    input_data = {'image':None,'depth':None,'height':None,'width':None,'calib':None,'label':None}
    image_path = self.data['image'][idx]
    calib_path = self.data['calib'][idx]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    height, width = image.shape[:2]
    image = self.transform_gen.get_transform(image).apply_image(image)
    input_data['image'] = image.transpose((2,0,1))
    input_data['height'] = height
    input_data['width'] = width

    input_data['calib'] = self.load_calib(calib_path)

    if self.mode == 'train':
      label_path = self.data['label'][idx]
      depth_path = self.data['depth'][idx]      
      input_data['label'] =  self.load_label(label_path)
      #the depth is pre-processed, so we just load it from png format
      depth = cv2.imread(depth_path)
      depth = cv2.cvtColor(depth,cv2.COLOR_BGR2RGB)
      depth = np.array(depth)
      depth = self.transform_gen.get_transform(depth).apply_image(depth)
      input_data['depth'] = depth.transpose((2,0,1))

    filtered = {k: v for k, v in input_data.items() if v is not None}
    input_data.clear()
    input_data.update(filtered)
    return input_data

  def load_calib(self,calib_path,cid = 2): #cid: camera id, 2 for the left color camera
    with open(calib_path, 'r') as f:
      C = f.readlines() 
    def parseLine(L, shape):
      data = L.split()
      data = np.array(data[1:]).reshape(shape).astype(np.float32)
      return data

    proj_c2p = parseLine(C[cid], shape=(3,4))
    filler = np.array([0, 0, 0, 1]).reshape((1,4))
    
    #The above gives us both proj_c2p and proj_v2c 
    intrinsics = proj_c2p[:3, :3] #This is our intrinsic parameters
    #Here is what the parameters are:
    #fx = intrinsics[0, 0]
    #fy = intrinsics[1, 1]
    #cx = intrinsics[0, 2]
    #cy = intrinsics[1, 2]
    return intrinsics 

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
    #verts, faces_idx, _ = load_obj(self.data[x])
    #faces = faces_idx.verts_idx
    # Initialize each vertex to be white in color.
    #verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    #textures = Textures(verts_rgb=verts_rgb.to(self.device))
    #mesh_obj= Meshes(verts=[verts.to(self.device)],faces=[faces.to(self.device)],textures=textures)
    #return mesh_obj
    mesh = load_objs_as_meshes(files = [self.data[x]],device = self.device)
    return mesh

  def __len__(self):
    return len(self.data)

