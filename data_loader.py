#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms

#Misc imports
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
from PIL import Image 

#Local imports
from utils import str_to_class


def get_dataset(name,mode,data_path,device):
  mode = 'train' if mode == 'val' else 'train'
  class_ = str_to_class(name,'data_loader')
  instance = class_(mode,data_path,device)
  return instance or None


class HIGHWAY(Dataset):
  def __init__(self,mode,data_path,device):
    self.mode = mode
    self.data_path = data_path
    self.device = device
    self.transform = transforms.Compose([
        transforms.Resize((375,1242), interpolation=2),
        transforms.ToTensor()])
    self.data = dict()

    image_path = os.path.join(data_path,'test/')
    image_samples = os.listdir(image_path)
    image_data = [os.path.join(data_path,'test/') + s for s in sorted(image_samples) if s.endswith('.png')]
    self.data['image'] = sorted(image_data)
  def __getitem__(self,idx):
    input_data = {'image':None}
    image_path = self.data['image'][idx]
    input_data['image'] = self.transform(Image.open(image_path).convert("RGB"))
    return input_data

  def __len__(self):
    return len(self.data['image'])

class KITTI(Dataset):
  def __init__(self,mode,data_path,device):
    self.mode = mode
    self.data_path = data_path
    self.device = device
    self.transform = transforms.Compose([
        transforms.Resize((375,1242), interpolation=2),
        transforms.ToTensor()])
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
    input_data['image'] = self.transform(Image.open(image_path).convert("RGB"))
    input_data['calib'] = self.load_calib(calib_path)

    if self.mode == 'train':
      label_path = self.data['label'][idx]
      depth_path = self.data['depth'][idx]      
      input_data['label'] =  self.load_label(label_path)
      #the depth is pre-processed, so we just load it from png format
      input_data['depth'] = self.transform(Image.open(depth_path))
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
