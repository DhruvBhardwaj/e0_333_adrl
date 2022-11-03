import torch
import numpy as np
import time
import datetime
import sys
import random
import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from config_1a_celeba import cfg
import utils as util
#from models import DiffusionNet
from ddpm_models import DiffusionNet, DiffusionClassifier
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
sys.stdout = util.Logger(cfg['training']['save_path'],'expt_1a_celeba_classifier.txt')
#########################################################3
#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
#########################################################3
classifier_cfg = cfg['classifier']

#########################################################3

def train():
    print('-' * 59)
    torch.cuda.empty_cache()
    model = DiffusionClassifier(cfg, classifier_cfg['num_classes'],device)

    model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=classifier_cfg['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    if(classifier_cfg['load_from_chkpt']):        
        chkpt_file = os.path.join(classifier_cfg['chkpt_path'],classifier_cfg['chkpt_file'])
        print('Loading checkpoint from:',chkpt_file)
        checkpoint = torch.load(chkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start=1+checkpoint['epoch']        
    else:
        epoch_start=1            
    
    model.train()
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(root=os.path.join(classifier_cfg['data_path'],'train'),transform=transform)
    val_dataset = ImageFolder(root=os.path.join(classifier_cfg['data_path'],'val'),transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=classifier_cfg['batch_size'],shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=classifier_cfg['batch_size'],shuffle=False)

    print('-' * 59)
    print("Starting Training of model")
    epoch_times = []

    for epoch in range(epoch_start,classifier_cfg['num_epochs']+1):        
        start_time = time.process_time()        
        total_loss = 0.0
        
        counter = 0
        for data in train_loader:
            image_batch, label = data            

            counter += 1            
            optimizer.zero_grad()           

            label_hat = model(image_batch.to(device)) 
                    
            loss = criterion(label_hat, label.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()            
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}/{}....... Loss={:12.5}"
                .format(epoch, counter, len(data), total_loss/classifier_cfg['batch_size']))
        
        current_time = time.process_time()
        print(N)
        print("Epoch {}/{} Done, Loss = {:12.5}"
                .format(epoch, classifier_cfg['num_epochs'], total_loss/N))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))        
        
        if(epoch%5==0):
            x = model.sample(cfg['ddpm']['image_size'],100,cfg['ddpm']['channels'])
            torch.save({
                'epoch': epoch,
                'loss':total_loss,                
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),                 
                }, os.path.join(classifier_cfg['chkpt_path'],'e' + str(epoch) + '_' + classifier_cfg['chkpt_file']))            
            
            
            util.save_image_to_file(epoch,0.5*(x[0]+1),classifier_cfg['save_path'],'TT_')
            util.save_image_to_file(epoch,0.5*(x[-1]+1),classifier_cfg['save_path'],'T0_')
            model.train()

        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return model

def sample_images_from_model(cfg,chkpt_file,num_samples, t_list=None):

    model = DiffusionNet(cfg, device)
    model.to(device)

    classifier_cfg=cfg['training']
    
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
    # util.save_image_to_file(000,0.5*(timed_samples+1),classifier_cfg['save_path'],'timed_samples_')