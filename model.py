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

  backbone = class_backbone() if class_backbone is not None else None
  pose_net = class_pose() if class_pose is not None else None
  shape_net = class_shape() if class_shape is not None else None

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

  def forward(self,im):
    if self.backbone is not None:
      backbone_x = self.backbone(im)
    #if self.pose_net is not None:
    #  pose_net_x = self.pose_net(x)
    #if self.shape_net is not None:
    # shape_net_x = self.shape_net(x)
    #if self.renderer is not None:
    # silhouete, image_ref = self.renderer.render(mesh)
    return backbone_x

  def visualize(self,inp,out,save_dir,idx):
    return self.backbone.visualize(inp,out,save_dir,idx)

class MaskRCNN(nn.Module):
  def __init__(self):
    super(MaskRCNN, self).__init__()
    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#    self.cfg.INPUT.FORMAT = 'RGB'
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(self.cfg)
    self.category_id = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck'}

  def forward(self,im):
    return self.predictor(im)

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


