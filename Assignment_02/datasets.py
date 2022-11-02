import torch
import numpy
import random
#import config_1a_celeba as config
import config_1a_bitmojis as config

from torchvision.io import ImageReadMode
import os, os.path
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self,img_folder, extn='.jpg'):
        self.img_folder=img_folder   
        self.extn = extn
        self.img_list = [name for name in os.listdir(self.img_folder) if name.endswith(self.extn)]
        #print(self.img_list[9])
        return
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,index):     
        #print(self.img_list[index])
        image=read_image(self.img_folder+'/'+self.img_list[index])    
        print(image.size())   
        #image = image[0,:,:].unsqueeze(0)   
        image=image/255.0        
        image=image.float()                
        image = -1.0 + 2.0*image
        return image

def getDataloader(data_path, batch_size, extn):
    print('[INFO] DATA_PATH={}, BATCH_SIZE={}'.format(data_path,batch_size))
    imgDataset = ImageDataset(data_path,extn)    
    print('[INFO] Found data set with {} samples'.format(len(imgDataset)))
    dl = DataLoader(imgDataset, batch_size,
                    shuffle=True)
    return dl, len(imgDataset)

if __name__ == '__main__':
    train_cfg = config.cfg['training']
    data, data_length = getDataloader(train_cfg['data_path'],train_cfg['batch_size'], train_cfg['file_extn'])
    for image_batch in data:        
        print(image_batch.size())
        print(torch.max(image_batch[0,:,:,:]),torch.min(image_batch[0,:,:,:]))
        print(torch.var(image_batch))
        break

