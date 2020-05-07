import torch
import torch.nn as nn

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from utils import str_to_class
import cv2
import os 
import numpy as np

def get_model(model,renderer,device = None): 
  class_backbone = str_to_class(model['backbone'],'model')
  class_pose = str_to_class(model['pose_net'],'model')
  class_shape = str_to_class(model['shape_net'],'model')

  backbone = class_backbone(device) if class_backbone is not None else None
  pose_net = class_pose(device) if class_pose is not None else None
  shape_net = class_shape(device) if class_shape is not None else None

  instance = Model(backbone,pose_net,shape_net,renderer,device)
  return instance

class Model(nn.Module):
  def __init__(self,backbone,pose_net,shape_net,renderer,device):
    super(Model, self).__init__()
    self.backbone = backbone
    self.pose_net = pose_net
    self.shape_net = shape_net
    self.renderer = renderer
    self.device = device

  def forward(self,im,K = [[721.5377,0.,609.5593],[0.,721.5377,172.854],[0.,0.,1.]]):
    K = torch.tensor(K).to(self.device)
    if self.backbone is not None:
      out = self.backbone(im,K)
    #if self.pose_net is not None:
    #  pose_net_x = self.pose_net(x)
    #if self.shape_net is not None:
    # shape_net_x = self.shape_net(x)
    if self.renderer is not None:
      out = self.renderer.render(mesh = im,K = K)
    return out

  def visualize(self,inp,out = None,save_dir = None,idx = None):
#    return self.renderer.visualize(inp,save_dir,idx)
    return self.backbone.visualize(inp,out,save_dir,idx)

class MaskRCNN(nn.Module):
  def __init__(self,device):
    super(MaskRCNN, self).__init__()
    self.device = device
    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(self.cfg)
    self.category_id = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck'}

  def forward(self,im,K_c):
    out = self.predictor(im)
    R_c = self.R_c_computation(K_c = K_c,c = out['instances'].pred_boxes.get_centers())

    K_r = self.K_r_computation(K_c = K_c, BROI = out['instances'].pred_boxes ,ROI = out['instances'].pred_boxes)
    H_inf = self.H_inf_computation(K_c = K_c,K_r = K_r, R_c = R_c)

#    K_c_ndc = self.intrinsics_to_ndc(K = K_c,image_h = im.shape[0],image_w = im.shape[1])

    return out

  def visualize(self,inp,out,save_dir = None,idx = None):
    v = Visualizer(inp[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(out["instances"].to("cpu"))
    if save_dir is not None:
      if not os.path.exists(os.path.join(save_dir,str(idx))):
        os.mkdir(os.path.join(save_dir,str(idx)))
      cv2.imwrite(os.path.join(save_dir,str(idx), 'full.png'),v.get_image()[:, :, ::-1])
      
      number_of_instances = [0]*20 #corresponding to pedestrian, car, bus, truck
      for i in range(out['instances'].pred_masks.shape[0]):
        class_id = out['instances'].pred_classes[i].item()
        if class_id in [0,2,5,7]: #pedestrian, car,bus,truck
          mask = 255 - out['instances'].pred_masks[i,:,:].squeeze(0).to('cpu').numpy()*255
          cv2.imwrite(os.path.join(save_dir,str(idx),self.category_id[class_id] +'_'+ str(number_of_instances[class_id])+'_instance.png'), np.float32(mask))
          number_of_instances[class_id] +=1

  def R_c_computation(self,K_c,c):
    '''
    batch_size: The number of detected object in the scene
    r^2 is considered as the matrix multiplication of r by itself (not multiplying each element of r to itself)
    c is the center of 2d bounding boxes on the image
    '''
    batch_size = c.shape[0]
    c = torch.cat((c,torch.ones(batch_size,1).to(self.device)),dim = 1).unsqueeze(2)
    K_c = K_c.unsqueeze(0).repeat([batch_size,1,1])
    K_c_inv = torch.inverse(K_c)
    q = torch.bmm(K_c_inv,c)
    p = torch.eye(3)[-1].unsqueeze(0).repeat([batch_size,1,1]).permute(0,2,1).to(self.device)
    
    r = torch.cross(p,q).squeeze()
    r_x = torch.zeros(batch_size,3,3).to(self.device)
    r_x[:,1,0] = -r[:,2]
    r_x[:,2,0] = r[:,1]
    r_x[:,0,1] = r[:,2]
    r_x[:,2,1] = -r[:,0]
    r_x[:,0,2] = -r[:,1]
    r_x[:,1,2] = r[:,0]
 
    I = torch.eye(3).unsqueeze(0).repeat([batch_size,1,1]).permute(0,2,1).to(self.device) 
    R_c = I + r_x + torch.bmm(r_x,r_x) /(torch.ones([batch_size,1,1]).to(self.device) + (p*q).sum(1,keepdim = True))
    return R_c

  def intrinsics_to_ndc(self,K,image_h,image_w):
    half_H = image_h/2
    half_W = image_w/2
    px_ndc = (K[0,2] - half_W) / half_W
    py_ndc = (K[1,2] - half_H) / half_H
    fx_ndc = K[0,0] / half_W
    fy_ndc = K[1,1] / half_H
    K_ndc = [[fx_ndc,0.,px_ndc],[0.,fy_ndc,py_ndc],[0.,0.,1.]]
    return K_ndc

  #adding batch support to this
  def K_r_computation(self,K_c, B_ROI, ROI): #B_ROI: object bbox before roi
    '''
    batch_size: The number of detected object in the scene
    
    '''
    print(B_ROI.shape)
    batch_size = B_ROI.shape[0]
    K_c = K_c.unsqueeze(0).repeat([batch_size,1,1])
    sy = ROI.shape[2]/B_ROI.shape[2] #height
    sx = ROI.shape[3]/B_ROI.shape[3] #width
    out = np.copy(intrinsics)
    out[0,0] *= sx
    out[0,2] = ROI.shape[3]*0.5#sx
    out[1,1] *= sy
    out[1,2] *= ROI.shape[2]*0.5
    return out

