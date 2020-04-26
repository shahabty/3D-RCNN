import yaml
import importlib

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

#batch size must be 1 because pytorch3d does not support multi-batch yet
def collate_fn(batch):
  if len(batch) == 1:
    return batch[0]
  return torch.stack([b for b in batch], 0)
