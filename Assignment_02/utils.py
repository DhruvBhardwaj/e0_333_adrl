import torch
import torchvision
import os
import datasets as DS
import random
from torchvision.utils import save_image
import torchvision.transforms as T
from torchvision.io import read_image
import sys

class Logger(object):
    def __init__(self, dir1='logs', filename="Default.log"):
        self.terminal = sys.stdout
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        self.log = open(os.path.join(dir1,filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass



def create_resampled_images():
    transform = T.Resize((64,64))

    dl = DS.getDataloader(cfg.DATA_PATH,cfg.BATCH_SIZE)
    k=0
    for image_batch in dl:
        print(image_batch.size())
        for i in range(0,image_batch.size(0)):
            save_image(transform(image_batch[i]),'./datasets/img_align_celeba_resampled/' + str(k) + '.jpg') 
            k +=1

    print(k)
    return

def save_image_to_file(epoch,image_tensor, save_path,ref_str=None):
    print(image_tensor.size())
    if ref_str is not None:
        filestr = save_path + ref_str +'SAMPLE_IMGS_E'+ str(epoch)  + '.jpg'
    else:
        filestr = save_path + 'SAMPLE_IMGS_E'+ str(epoch)  + '.jpg'
    save_image(image_tensor,filestr,nrow = 10) 
    return

def return_random_batch_from_dir(img_folder, file_extn, num_samples):
    img_list = [name for name in os.listdir(img_folder) if name.endswith(file_extn)]
    samples=[]
    if(len(img_list)>0):
        
        sample_names = random.sample(img_list, num_samples)
        for name in sample_names:
            img = read_image(img_folder+'/'+name).float()
            img = img/255.0
            samples.append((img.unsqueeze(0)))
        samples = torch.cat(samples)
        print(samples.size())
    return samples

#save_image_to_file(0,torch.randn(100,3,64,64))
#create_resampled_images()
#return_random_batch_from_dir('./datasets/tiny_imagenet/', '.JPEG', 10)