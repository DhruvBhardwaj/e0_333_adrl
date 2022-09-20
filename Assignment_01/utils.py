import torch
import torchvision
import os
import datasets as DS
import random
from torchvision.utils import save_image
import torchvision.transforms as T
from torchvision.io import read_image

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

def save_image_to_file(epoch,image_tensor, save_path):
    print(image_tensor.size())
    save_image(image_tensor,save_path + 'SAMPLE_IMGS_E'+ str(epoch)  + '.jpg',nrow = 10) 
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