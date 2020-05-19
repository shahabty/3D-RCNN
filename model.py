import torch
import torch.nn as nn

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


from rcnn import RCNN

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

  def forward(self,input_data):
    if self.backbone is not None: 
      out_batch = self.backbone(input_data)
      H_inf = self.H_inf_computation(out_batch = out_batch,K_c_batch = input_data['calib'].to(self.device))
      return H_inf
    #if self.pose_net is not None:
    #  pose_net_x = self.pose_net(x)
    #if self.shape_net is not None:
    # shape_net_x = self.shape_net(x)
    #if self.renderer is not None:
    #  K_c_ndc = self.K_to_ndc(K = K_c,image_h = im.shape[0],image_w = im.shape[1])
    #  out = self.renderer.render(mesh = im,K = K_c_ndc)
    #return out
    
  def H_inf_computation(self,out_batch,K_c_batch):
    H_inf = []
    for out,K_c in zip(out_batch,K_c_batch):
      K_c = K_c.unsqueeze(0).repeat([out['instances'].pred_boxes.tensor.shape[0],1,1])
      R_c = self.R_c_computation(K_c = K_c,c = out['instances'].pred_boxes.get_centers())
      K_r = self.K_r_computation(K_c = K_c, B_ROI = out['instances'].pred_boxes ,ROI = out['instances'].pred_boxes)
      H_inf.append(torch.bmm(K_r,torch.bmm(torch.inverse(R_c),torch.inverse(K_c))))
    return H_inf

  def R_c_computation(self,K_c,c):
    '''
    batch_size: The number of detected object in the scene
    r^2 is considered as the matrix multiplication of r by itself (not multiplying each element of r to itself)
    c is the center of 2d bounding boxes on the image
    '''
    batch_size = c.shape[0]
    c = torch.cat((c,torch.ones(batch_size,1).to(self.device)),dim = 1).unsqueeze(2)
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

  def K_r_computation(self,K_c, B_ROI, ROI): #B_ROI: object bbox before roi
    '''
    batch_size: The number of detected objects in the scene
    '''
    B_ROI = B_ROI.tensor
    ROI = ROI.tensor
    batch_size = B_ROI.shape[0]
    B_ROI_H = B_ROI[:,3]- B_ROI[:,1]
    B_ROI_W = B_ROI[:,2]- B_ROI[:,0]
    ROI_H =ROI[:,3]- ROI[:,1]
    ROI_W =ROI[:,2]- ROI[:,0]
    sy = (ROI_H / B_ROI_H)
    sx = (ROI_W / B_ROI_W)
    out = K_c.clone()
    out[:,0,0] *= sx
    out[:,0,2] = ROI_W*0.5
    out[:,1,1] *= sy
    out[:,1,2] *= ROI_H*0.5
    return out

  def K_to_ndc(self,K,image_h,image_w):
    half_H = image_h/2
    half_W = image_w/2
    px_ndc = (K[0,2] - half_W) / half_W
    py_ndc = (K[1,2] - half_H) / half_H
    fx_ndc = K[0,0] / half_W
    fy_ndc = K[1,1] / half_H
    K_ndc = [[fx_ndc,0.,px_ndc],[0.,fy_ndc,py_ndc],[0.,0.,1.]]
    return K_ndc

class MaskRCNN(nn.Module):
  def __init__(self,device):
    super(MaskRCNN, self).__init__()
    self.device = device
    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.META_ARCHITECTURE = 'RCNN'
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.cfg_clone = self.cfg.clone()  # cfg can be modified by model
    self.model = build_model(self.cfg_clone)
    self.model.eval()
    self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(self.cfg.MODEL.WEIGHTS)
    self.input_format = self.cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format

    self.category_id = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck'}

  def forward(self,input_data):
    return self.inference(input_data)

  def inference(self,input_data):
    with torch.no_grad():
      predictions = self.model(input_data)[0]
      return predictions

  def visualize(self,input_data_batch,out_batch,save_dir = None,idx = None):
    image_batch = input_data_batch['image']
    for j,(image,out) in enumerate(zip(image_batch,out_batch)):
      v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
      v = v.draw_instance_predictions(out["instances"].to("cpu"))
      if save_dir is not None:
        if not os.path.exists(os.path.join(save_dir,str(idx))):
          os.mkdir(os.path.join(save_dir,str(idx + j)))
        cv2.imwrite(os.path.join(save_dir,str(idx + j), 'full.png'),v.get_image()[:, :, ::-1])
      
        number_of_instances = [0]*20 #corresponding to pedestrian, car, bus, truck
        for i in range(out['instances'].pred_masks.shape[0]):
          class_id = out['instances'].pred_classes[i].item()
          if class_id in [0,2,5,7]: #pedestrian, car,bus,truck
            mask = 255 - out['instances'].pred_masks[i,:,:].squeeze(0).to('cpu').numpy()*255
            cv2.imwrite(os.path.join(save_dir,str(idx + j),self.category_id[class_id] +'_'+ str(number_of_instances[class_id])+'_instance.png'), np.float32(mask))
            number_of_instances[class_id] +=1
