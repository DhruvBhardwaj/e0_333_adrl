import pandas
import os
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.utils import save_image

data='/home/dhruvb/adrl/Assignment_01/datasets/img_align_celeba/'
out_data='/home/dhruvb/adrl/datasets/img_align_celeba_classes/'
labels='/home/dhruvb/adrl/datasets/celeba/list_attr_celeba.xlsx'

df = pandas.read_excel(labels, header=0)
attr_list=[
    'Mustache','Bald','Eyeglasses','Wearing_Hat',	'Wearing_Necktie','Black_Hair','Blond_Hair','No_Beard','Smiling',	'Male'
]
print(df.head())
transform = T.Resize((64,64))

for attr in attr_list:
    print('Processing:',attr)
    directory = os.path.join(out_data,attr)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df2=df.loc[df[attr]==1,'None']
    df=df.drop(df[df[attr]==1].index)
    print(len(df2))
    for fname in df2:
        #print(os.path.join(data,fname))        
        image=read_image(os.path.join(data,fname))
        image=image/255.0        
        image=image.float()
        save_image(transform(image),os.path.join(directory,fname))        

