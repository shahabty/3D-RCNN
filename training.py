from utils import str_to_class
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

def get_trainer(train_loader,val_loader,model,loss_function,optimizer,save_dir,device):
  class_ = str_to_class('Trainer','training')
  instance = class_(train_loader,val_loader,model,loss_function,optimizer,save_dir,device)
  return instance or None

class Trainer():
  def __init__(self,train_loader,val_loader,model,loss_function,optimizer,save_dir,device):
    self.device = device
    self.save_dir = save_dir
    self.train_loader = train_loader
    self.val_lodaer = val_loader
    self.model = model
    self.losses = loss_function
    self.optimizer = optimizer
    self.writer = SummaryWriter()

  def run(self):
    for idx,input_data in tqdm(enumerate(self.train_loader)):
#      x = vutils.make_grid(input_data[0]['image'].data.cpu(), normalize=False, scale_each=True)
#      self.writer.add_image('Image', x, idx)

      out = self.model(input_data)

      #setup the camera in forward call
      #silhouete = self.model(input_data)      
      #self.model.renderer.visualize(inp = silhouete,save_dir = self.save_dir,idx = idx) 

      self.model.backbone.visualize(input_data,out,self.save_dir,idx)



      #self.renderer.visualize(silhouete,image_ref,save_dir = 'rendered',idx = idx)

