import os
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.utils import save_image

gan_in = '/home/dhruvb/adrl/datasets/bitmojis_resampled/'
#'/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/images/bitmoji_gan_output'
gan_out = '/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/images/bitmojis_original/'
#'/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/images/bitmoji_gan_output_resampled'

img_list = [name for name in os.listdir(gan_in) if name.endswith('.png')]
print(len(img_list))
#transform = T.Resize((64,64))

for i in range(1000):
    image=read_image(os.path.join(gan_in,img_list[i]))    
    image=image/255.0        
    image=image.float()
    #save_image(transform(image),os.path.join(gan_out,img))
    save_image(image,os.path.join(gan_out,img_list[i]))

