from utils import str_to_class


def get_trainer(train_loader,val_loader,model,renderer,loss_function,optimizer):
  class_ = str_to_class('Trainer','training')
  instance = class_(train_loader,val_loader,model,renderer,loss_function,optimizer)
  return instance or None

class Trainer():
  def __init__(self,train_loader,val_loader,model,renderer,loss_function,optimizer):
    self.train_loader = train_loader
    self.val_lodaer = val_loader
    self.model = model
    self.renderer = renderer
    self.losses = loss_function
    self.optimizer = optimizer
  def run(self):
    pass
