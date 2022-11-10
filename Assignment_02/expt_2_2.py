import torch
import numpy as np
import time
import datetime
import sys
import random
import os

from config_1a_bitmojis import cfg
from datasets import getDataloader
import utils as util

from ddpm_models import EBM
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
sys.stdout = util.Logger(cfg['training']['save_path'],'expt_2_bitmojis.txt')
#########################################################3
#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
#########################################################3
train_cfg = cfg['training']
def train():
    print('-' * 59)    
    model = EBM(cfg, device)

    model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    epoch_start=1            
    
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
            image_batch_samples = model.sample(image_batch.size())                       
            #image_batch_samples = model.sample(image_batch.to(device))
            optimizer.zero_grad()                                  
            
            es = model(image_batch_samples.to(device))                                                     
            e = model(image_batch.to(device)) 

            loss = model.criterion(es.detach(), e)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            
            total_loss += loss.item()            
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}/{}....... Loss={:12.5}"
                .format(epoch, counter, len(data), total_loss/train_cfg['batch_size']))                                
            
            if counter%1000 == 0:                
                break
                
        
        current_time = time.process_time()
        print(N)
        print("Epoch {}/{} Done, Loss = {:12.5}"
                .format(epoch, train_cfg['num_epochs'], total_loss/N))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))        
        
        if(epoch%1==0):
            model.eval()
            x = model.sample_from_buffer((100,3,64,64))
            print(x.size())
            # torch.save({
            #     'epoch': epoch,
            #     'loss':total_loss,                
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),                 
            #     }, os.path.join(train_cfg['chkpt_path'],'e' + str(epoch) + '_' + train_cfg['chkpt_file']))            
            util.save_image_to_file(epoch,0.5*(x+1),train_cfg['save_path'],str(cfg['ebm']['num_steps'])+'EBM_')
            model.train()

        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return model

def sample_images_from_model(cfg,chkpt_file,num_samples, t_list=None):

    model = DiffusionNet(cfg, device)
    model.to(device)

    train_cfg=cfg['training']
    
    print('Loading checkpoint from:',chkpt_file)
    checkpoint = torch.load(chkpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    x = model.sample(cfg['ddpm']['image_size'],num_samples,cfg['ddpm']['channels'])
    print(len(x))
    print(x[0].size())
    timed_samples=[]
    if(t_list is not None):        
        for t in t_list:
            timed_samples.append(x[t])
    print(len(timed_samples))
    print(timed_samples[0].size())
    return x, timed_samples

if __name__ == '__main__':
    print(cfg)
    model = train()

    # chkpt_file = '/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/bitmoji/e15_expt_1a_bitmojis.chk.pt'
    # x,timed_samples = sample_images_from_model(cfg,chkpt_file,10,[i for i in range(0,500,49)])
    # timed_samples=torch.cat(timed_samples,dim=0)
    # print(timed_samples.size())
    # util.save_image_to_file(000,0.5*(timed_samples+1),train_cfg['save_path'],'timed_samples_')