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

#batch size more than 1 must be implemented here
def collate_fn_3D(batch):
  if len(batch) == 1:
    return batch[0]
  return torch.stack([b for b in batch], 0)

def collate_fn_maskrcnn(batch):
  for b in batch:
    filtered = {k: v for k, v in b.items() if v is not None}
    b.clear()
    b.update(filtered)
  return dataloader.default_collate(batch)

