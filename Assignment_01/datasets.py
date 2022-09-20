import torch
import numpy
import random
import config_tiny_imagenet as cfg

import os, os.path
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    return 

g = torch.Generator()
g.manual_seed(42)

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
        image=image.float()                
        image=image/255.0        
        return image

def getDataloader(data_path, batch_size, extn):
    print('[INFO] DATA_PATH={}, BATCH_SIZE={}'.format(data_path,batch_size))
    imgDataset = ImageDataset(data_path,extn)    
    print('[INFO] Found data set with {} samples'.format(len(imgDataset)))
    dl = DataLoader(imgDataset, batch_size,
                    shuffle=True,worker_init_fn=seed_worker,generator=g)
    return dl

if __name__ == '__main__':
    print(cfg.DATA_PATH)
    data = getDataloader(cfg.DATA_PATH, cfg.BATCH_SIZE, cfg.FILE_EXTN)
    for image_batch in data:        
        print(image_batch.size())
        print(torch.var(image_batch))

