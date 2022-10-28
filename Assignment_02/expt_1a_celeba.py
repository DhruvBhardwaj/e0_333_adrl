import torch
import numpy as np
import time
import datetime
import sys
import random
import os

from config_1a_celeba import cfg
from datasets import getDataloader
import utils as util
from models import DiffusionNet
#########################################################3
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
#########################################################3
#########################################################3
#cfg = config.cfg
#########################LOGGER#########################
sys.stdout = util.Logger(cfg['training']['save_path'],'expt_1a_celeba.txt')
#########################################################3

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
#########################################################3
train_cfg = cfg['training']
def train():
    print('-' * 59)
    torch.cuda.empty_cache()
    model = DiffusionNet(cfg, device)

    model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    if(train_cfg['load_from_chkpt']):        
        chkpt_file = os.path.join(train_cfg['chkpt_path'],train_cfg['chkpt_file'])
        print('Loading checkpoint from:',chkpt_file)
        checkpoint = torch.load(chkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start=1+checkpoint['epoch']        
    else:
        epoch_start=1        

    model.eval()
    x = model.sample(N=2,end_T=1)
    util.save_image_to_file(epoch,0.5*(x+1),train_cfg['save_path'])
    print(aaa)
            
    model.train()

    data, N = getDataloader(train_cfg['data_path'],train_cfg['batch_size'], train_cfg['file_extn'])
    
    print('-' * 59)
    print("Starting Training of model")
    epoch_times = []

    for epoch in range(epoch_start,train_cfg['num_epochs']+1):        
        start_time = time.process_time()        
        total_loss = 0.0
        
        counter = 0
        for image_batch in data:
            
            counter += 1            
            optimizer.zero_grad()           
            
            e_hat, e = model.forward(image_batch.to(device)) 
                    
            loss = model.criterion(image_batch.to(device), e_hat, e)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()            
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}/{}....... Loss={:12.5}"
                .format(epoch, counter, len(data), total_loss/train_cfg['batch_size']))
        
        current_time = time.process_time()
        print(N)
        print("Epoch {}/{} Done, Loss = {:12.5}"
                .format(epoch, train_cfg['num_epochs'], total_loss/N))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))        
        
        if(epoch%10==0):
            torch.save({
                'epoch': epoch,
                'loss':total_loss,                
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),                
                }, os.path.join(train_cfg['chkpt_path'],train_cfg['chkpt_file']))
            model.eval()
            x = model.sample(N=2,end_T=1)
            util.save_image_to_file(epoch,0.5*(x+1),train_cfg['save_path'])
            model.train()

        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return model

if __name__ == '__main__':
    model = train()