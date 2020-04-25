import yaml


def load_config(path):
  with open(path, 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

  return cfg

