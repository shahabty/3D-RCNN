from utils import str_to_class
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


def get_tester(test_loader,model,save_dir,device):
  class_ = str_to_class('Tester','testing')
  instance = class_(test_loader,model,save_dir,device)
  return instance or None

class Tester():
  def __init__(self,test_loader,model,save_dir,device):
    self.device = device
    self.save_dir = save_dir
    self.test_loader = test_loader
    self.model = model
    self.writer = SummaryWriter()
  def run(self):
    for idx, input_data in tqdm(enumerate(self.test_loader)):
      out = self.model(input_data)
      self.model.backbone.visualize(input_data,out,self.save_dir,idx)

      #self.renderer.visualize(silhouete,image_ref,save_dir = 'rendered',idx = idx)
