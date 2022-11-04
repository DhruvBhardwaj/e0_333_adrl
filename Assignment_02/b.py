import torch
x = [torch.randn(10,3,64,64)]
x = torch.cat(x)
print(x.size())