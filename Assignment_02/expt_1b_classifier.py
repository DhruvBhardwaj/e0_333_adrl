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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
#########################################################3
classifier_cfg = cfg['classifier']

#########################################################3
def validate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    for data in valid_loader:
        image_batch, label = data                  
        t = torch.randint(low=0,high=cfg['diffusion']['T']-1,size=(image_batch.size(0),),device=device).long()
        label_hat = model(image_batch.to(device),t) 
                    
        loss = criterion(label_hat, label.to(device))        
            
        total_loss += loss.item() 
    
    model.train()
    return total_loss

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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        ])

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
            t = torch.randint(low=0,high=cfg['diffusion']['T']-1,size=(image_batch.size(0),),device=device).long()

            label_hat = model(image_batch.to(device),t) 
                    
            loss = criterion(label_hat, label.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()            
            
            if counter%500 == 0:                
                print("Epoch {}......Step: {}....... Loss={:12.5}"
                .format(epoch, counter, total_loss))
        
        current_time = time.process_time()        
        print("Epoch {}/{} Done, Loss = {:12.5}"
                .format(epoch, classifier_cfg['num_epochs'], total_loss))
        val_loss = validate(model, valid_loader, criterion)
        print("Epoch {}/{} Done, Val Loss = {:12.5}"
                .format(epoch, classifier_cfg['num_epochs'], val_loss))

        print("Total Time Elapsed={:12.5} seconds".format(str(current_time-start_time)))        
        
        if(epoch%5==0):            
            torch.save({
                'epoch': epoch,
                'loss':total_loss,                
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),                 
                }, os.path.join(classifier_cfg['chkpt_path'],'classifier' + str(epoch) + '_' + classifier_cfg['chkpt_file']))            

        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:12.5} seconds".format(str(sum(epoch_times))))
    return model

if __name__ == '__main__':
    print(cfg)
    #model = train()

    chkpt_file = '/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/celeba/e15_expt_1a_celeba.chk.pt'
    model = DiffusionNet(cfg, device)
    model.to(device)    
    
    print('Loading diffusion checkpoint from:',chkpt_file)
    checkpoint = torch.load(chkpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval()

    if(cfg['diffusion']['guided']):                    
        classifier = DiffusionClassifier(cfg, cfg['classifier']['num_classes'],device)
        print('Loading classifier from:',cfg['diffusion']['guiding_classifier'])
        checkpoint = torch.load(cfg['diffusion']['guiding_classifier'])
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)
        classifier.eval()

    x=[]
    for i in range(10):
        x0 = model.sample(cfg['ddpm']['image_size'],cfg['classifier']['num_classes'],cfg['ddpm']['channels'], classifier)             
        x.append(x0[-1])
    
    x=torch.cat(x)
    print(x.size())
    util.save_image_to_file(0,0.5*(x+1),cfg['training']['save_path'],'Conditional_T0_')
    