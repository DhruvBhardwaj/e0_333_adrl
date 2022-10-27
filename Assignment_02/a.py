import torch

# x = torch.tensor([[1,1,1],[2,2,2]]).permute(1,0)
# print(x.size())

# w = torch.tensor([[3,3]])
# print(w.size())

# y = torch.mul(w,x)
# print(y)
# print(y.size())

# x = torch.tensor([[1,2,3,4,2,6,7]])
# y = (x == 22).nonzero(as_tuple=False)
# print(y)
# print(y.numel())
# print(x.size())

# z = torch.randn(50,100)
# print(z[y[:,1]].size())

# aa = torch.randn((128,1200))
# z = 0
# k = aa + z
# print(k.size())

from config_1a_celeba import cfg
from datasets import getDataloader
import utils as util
from models import DiffusionNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
chkpt_file = '/home/dhruvb/adrl/e0_333_adrl/Assignment_02/chkpt/expt_1a_celeba.chk.pt'
print('Loading checkpoint from:',chkpt_file)
checkpoint = torch.load(chkpt_file)

model = DiffusionNet(cfg, device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

s = model.sample(2)
print(len(s))

        