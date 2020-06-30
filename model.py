#Pytorch imports
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.jit.annotations import Tuple, List, Dict, Optional

#Misc imports
from utils import str_to_class
import os 
import numpy as np
import cv2

#imports
from visualization import select_top_predictions, overlay_boxes,overlay_mask,overlay_keypoints, overlay_class_names

def get_model(renderer,cfg,mode,device = None):
  model = cfg[mode]['model']
  class_backbone = str_to_class(model['backbone'],'model')
  class_pose = str_to_class(model['pose_net'],'model')
  class_shape = str_to_class(model['shape_net'],'model')

  backbone = class_backbone(cfg,mode,device) if class_backbone is not None else None
  pose_net = class_pose(cfg,mode,device) if class_pose is not None else None
  shape_net = class_shape(cfg,mode,device) if class_shape is not None else None

  instance = Model(backbone,pose_net,shape_net,renderer,mode,device)
  return instance

class Model(nn.Module):
  def __init__(self,backbone,pose_net,shape_net,renderer,mode,device):
    super(Model, self).__init__()
    self.backbone = backbone.eval() if mode != 'train'and backbone != None else backbone
    self.pose_net = pose_net.eval() if mode != 'train' and pose_net != None else pose_net
    self.shape_net = shape_net.eval() if mode != 'train' and shape_net != None else shape_net
    self.renderer = renderer
    self.mode = mode
    self.device = device

  def forward(self,input_data):
    input_data = {k:v.to(self.device) for k,v in input_data.items()}
    if self.backbone is not None: 
      features,bboxes = self.backbone(input_data)
      #H_inf = self.H_inf_computation(out_batch = out_batch,input_data_batch = input_data)
      #return output
    if self.pose_net is not None:
      pose_net_x = self.pose_net(features,input_data)
    if self.shape_net is not None:
     shape_net_x = self.shape_net(x)
    if self.renderer is not None:
      K_c_ndc = self.K_to_ndc(K = K_c,image_h = im.shape[0],image_w = im.shape[1])
      out = self.renderer.render(mesh = im,K = K_c_ndc)
    return out
    
  def H_inf_computation(self,out_batch,input_data_batch):
    H_inf = []
    for out,input_data in zip(out_batch,input_data_batch):
      K_c = input_data['calib'].to(self.device)
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

  def backbone_visualization(self,input_data,predicted,is_mask = True,is_bbox = True,is_keypoint = False,save_dir = None,idx = None):
    for img,result in zip(input_data['image'].split(1),predicted):
      img = img.squeeze()
      self.backbone.visualization(img,result,is_mask,is_bbox,is_keypoint,save_dir,idx)


class ShapeNet(nn.Module):
  def __init__(self,cfg,mode,device):
    self.cfg = cfg
    self.device = device
    self.mode = mode
    self.nn = 
    self.pca = h5py.File(,'r')['tsdf_basis'][()]

  def forward(self,object_feature,input_data):
    





class MaskRCNN(nn.Module):
  def __init__(self,cfg,mode,device):
    super(MaskRCNN, self).__init__()
    self.cfg = cfg
    self.device = device
    self.mode = mode
    if cfg[mode]['backbone']['pretrained'] is None:
      model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
      model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
      self.load_weights(model)

    self.transform = model.transform
    self.feature_extractor = model.backbone #this is the original feature maps extracted by the backbone
    self.rpn = model.rpn #this is the predicted bounding boxes before ROIalign
    self.box_roi_pool = model.roi_heads.box_roi_pool
    self.roi_heads = model.roi_heads #these are our outputs
#    print(self.roi_heads)
  def forward(self,input_data):
    if self.mode == 'train':
      images,targets = input_data['image'],input_data['target']
    else:
      images,targets = input_data['image'], None

    if self.training and targets is None:
      raise ValueError("In training mode, targets should be passed")
    if self.training:
      assert targets is not None
      for target in targets:
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
          if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
            raise ValueError("Expected target boxes to be a tensor of shape [N, 4], got {:}.".format(boxes.shape))
        else:
          raise ValueError("Expected target boxes to be of type Tensor, got {:}.".format(type(boxes)))

    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
      val = img.shape[-2:]
      assert len(val) == 2
      original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
      for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
          # print the first degenrate box
          bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
          degen_bb: List[float] = boxes[bb_idx].tolist()
          raise ValueError("All bounding boxes should have positive height and width.Found invaid box {} for target at index {}.".format(degen_bb, target_idx))

    features = self.feature_extractor(images.tensors)
    if isinstance(features, torch.Tensor):
      features = OrderedDict([('0', features)])
    proposals, proposal_losses = self.rpn(images, features, targets)
    pre_roi_proposals = proposals
    post_roi_proposals = self.box_roi_pool(features,proposals,images.image_sizes)
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    pre_roi_proposals = torch.cat(pre_roi_proposals)
    #print(pre_roi_proposals[0].shape)
    #print(post_roi_proposals[0].shape)
    #print(detections)
    if self.mode == 'train':
      losses = {}
      losses.update(detector_losses)
      losses.update(proposal_losses)
      return losses
    return detections

  def visualization(self,img,result,is_mask,is_bbox,is_keypoint,save_dir,idx):
    top_predictions = select_top_predictions(result, 0.7)
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
    img = np.array(img.permute(1,2,0)*255)[:, :, [2, 1, 0]]
    vis_img = img.copy()
    if is_bbox:
      vis_img = overlay_boxes(vis_img, top_predictions)
    if is_mask:
      vis_img = overlay_mask(vis_img, top_predictions)
    if is_keypoint:
      vis_img = overlay_keypoints(vis_img, top_predictions)
    
    vis_img = overlay_class_names(vis_img, top_predictions)
    cv2.imwrite(os.path.join(save_dir,str(idx).zfill(4) +'.png'), np.float32(vis_img))


class PoseNet(nn.Module):
  def __init__(self,cfg,mode,device):
    super(PoseNet, self).__init__()
    self.cfg = cfg
    self.mode = mode
    self.device = device
  def forward(self,features):
    pass    

