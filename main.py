import torch
from torch.utils.data import DataLoader

from utils import load_config
from data_loader import get_dataset

cfg = load_config('config.yaml')
is_train = cfg['mode']['train']
is_val = cfg['mode']['val']
is_test = cfg['mode']['test']
mode = 'train' if is_train or is_val else 'test'
device = torch.device("cuda:0" if (torch.cuda.is_available() and not cfg[mode]['no_cuda']) else "cpu")
torch.cuda.set_device(device)

if __name__ == '__main__':
  if mode == 'train':
    train_dataset = get_dataset(name = 'Cars',mode = 'train',data_path = cfg['data']['data_path'],device = device)
    val_dataset = get_dataset(name = 'Cars',mode = 'val',data_path = cfg['data']['data_path'],device = device) 
    train_loader = DataLoader(train_dataset,batch_size=cfg['train']['batch_size'], num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], num_workers=8, shuffle=True)
  else:
    test_dataset = get_dataset(name = 'Cars',mode = 'test',data_path = cfg['data']['data_path'],device = device)
    test_loader = DataLoader(test_dataset,batch_size=cfg['test']['batch_size'], num_workers=8, shuffle=True)

  if mode == 'train':
    trainer


