from utils import str_to_class
from tqdm import tqdm

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
  def run(self):
    for idx,input_data in tqdm(enumerate(self.train_loader)):
      im = input_data['image'].squeeze(0).numpy()
      depth = input_data['depth'].squeeze(0).numpy()
#      calib_matrix = input_data['calib']
#      label = input_data['label']
      out = self.model(im)
      self.model.visualize(im,out,self.save_dir,idx)
      #self.renderer.visualize(silhouete,image_ref,save_dir = 'rendered',idx = idx)

