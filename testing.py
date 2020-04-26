from utils import str_to_class

def get_tester(test_loader,model,device):
  class_ = str_to_class('Tester','testing')
  instance = class_(test_loader,model,device)
  return instance or None

class Tester():
  def __init__(self,test_loader,model,device):
    self.device = device
    self.test_loader = test_loader
    self.model = model

  def run(self):
    for idx, inp in enumerate(self.test_loader):
      self.model(inp)
      #self.renderer.visualize(silhouete,image_ref,save_dir = 'rendered',idx = idx)
