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
      backbone_x = self.backbone(x)
    #if self.pose_net is not None:
    #  pose_net_x = self.pose_net(x)
    #if self.shape_net is not None:
    # shape_net_x = self.shape_net(x)
    #if self.renderer is not None:
    # silhouete, image_ref = self.renderer.render(mesh)

class MaskRCNN(nn.Module):
  def __init__(self):
    super(MaskRCNN, self).__init__()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(cfg)

  def forward(self,im):
    output = self.predictor(im)
    print(output.keys)


  def visualize(self,inp,out,save_dir = None,idx = None):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    if save_dir is not None:
      cv2_imwrite(os.path.join(save_dir,str(idx) + '.png'),v.get_image()[:, :, ::-1])


