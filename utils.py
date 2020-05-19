import torch
import yaml
import importlib
import logging
from torch.utils.data import dataloader

def load_config(path):
  with open(path, 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

  return cfg

def str_to_class(name,module_name):
  if name == None:
    return None
  try:
    module = importlib.import_module(module_name)
    try:
      class_ = getattr(module, name)
    except AttributeError:
      logging.error('The dataset is not implemented!')
  except ImportError:
    logging.error('Module does not exist!')
  return class_ or None

class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)




#batch size more than 1 must be implemented here
#def collate_fn_3D(batch):
#  if len(batch) == 1:
#    return batch[0]
#  return torch.stack([b for b in batch], 0)

#def collate_fn_maskrcnn(batch):
#  for k,v in batch[0].items():
#    print(v.shape)
#  for k,v in batch[1].items():
#    print(v.shape)  
#  batch = dataloader.default_collate(batch)
#  batch = [batch[k] = batch[k].squeeze().numpy() for k,v in batch]
#  return batch
