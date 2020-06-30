#Pytorch imports
import torch
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method('spawn')

#Local imports
from utils import load_config
from data_loader import get_dataset
from model import get_model
from render import get_renderer
from training import get_trainer
from testing import get_tester

cfg = load_config('config.yaml')
is_train = cfg['mode']['train']
is_val = cfg['mode']['val']
is_test = cfg['mode']['test']
mode = 'train' if is_train or is_val else 'test'
device = torch.device("cuda:0" if (torch.cuda.is_available() and not cfg[mode]['no_cuda']) else "cpu")
torch.cuda.set_device(device)

if __name__ == '__main__':
  if mode == 'train':
    train_dataset = get_dataset(name = cfg['data']['dataset'], mode = 'train',data_path = cfg['data']['data_path'],device = device)
    val_dataset = get_dataset(name = cfg['data']['dataset'], mode = 'val',data_path = cfg['data']['data_path'],device = device)
    train_loader = DataLoader(train_dataset,batch_size=cfg['train']['batch_size'],num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'],num_workers=4, shuffle=True)
    #The renderer is implemented separately. Becuase it is replacable and it is not present in inference.
    renderer = get_renderer(name = cfg['train']['model']['renderer'],device = device)
    model = get_model(renderer = renderer,cfg = cfg,mode = mode, device = device).to(device)

    trainer = get_trainer(train_loader = train_loader,val_loader = val_loader, model = model,loss_function = cfg['train']['loss'], optimizer = cfg['train']['optimizer'],save_dir = cfg['data']['save_dir'],device = device)
    trainer.run()
  else:
    test_dataset = get_dataset(name = cfg['data']['dataset'],mode = 'test',data_path = cfg['data']['data_path'],device = device)
    test_loader = DataLoader(test_dataset,batch_size=cfg['test']['batch_size'],num_workers=4, shuffle=False)

    renderer = get_renderer(name = cfg['test']['model']['renderer'],device = device)
    model = get_model(cfg = cfg,renderer = renderer,mode = mode,device = device).to(device)
    tester = get_tester(test_loader = test_loader,model = model,save_dir = cfg['data']['save_dir'],device = device)
    tester.run()
