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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
#########################################################3
train_cfg = cfg['training']
def train():
    print('-' * 59)
    torch.cuda.empty_cache()
    model = DiffusionNet(cfg, device)

    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])

    data, N = getDataloader(train_cfg['data_path'],train_cfg['batch_size'], train_cfg['file_extn'])
    
    print('-' * 59)
    print("Starting Training of model")
    epoch_times = []

    for epoch in range(1,train_cfg['num_epochs']+1):        
        start_time = time.process_time()        
        total_loss = 0.0
        lxt_xt1 = 0.0
        lx0_x1 = 0.0
        counter = 0
        for image_batch in data:
            
            counter += 1            
            optimizer.zero_grad()           
            
            e_hat, e = model.forward(image_batch.to(device)) 
                    
            loss, lxt, lx0 = model.criterion(image_batch.to(device), e_hat, e)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            lxt_xt1 += lxt.item()
            lx0_x1 += lx0.item()
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}/{}....... Loss={:12.5} (l[xt<-xt1]={:12.5},l[x0<-x1]={:12.5})"
                .format(epoch, counter, len(data), total_loss/train_cfg['batch_size'],
                lxt_xt1/train_cfg['batch_size'],lx0_x1/train_cfg['batch_size']))
        
        current_time = time.process_time()
        print(N)
        print("Epoch {}/{} Done, Loss = {:12.5} (l[xt<-xt1]={:12.5},l[x0<-x1]={:12.5})"
                .format(epoch, train_cfg['num_epochs'], total_loss/N,
                lxt_xt1/N,lx0_x1/N))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))        
        
        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return model

if __name__ == '__main__':
    model = train()