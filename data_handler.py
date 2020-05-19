import torch
import numpy as np
from utils import Singleton

@Singleton
class Data:
  def __init__(self):
    pass
  def from_batch(self,input_batch):
    '''
    Detectron2 requires list of dictionaries in numpy. Each dictionary represents a sample
    '''
#    for k,v in input_batch.items():
#      v = v.squeeze()

#      if k != 'calib':
#        v = v.numpy()
#      input_batch[k] = v
#    return input_batch
  #todo
  def from_list(self,input_list):
    '''
    pytorch3d handles batches in Pytorch Tensors
    '''
    return torch.FloatTensor(input_list)
