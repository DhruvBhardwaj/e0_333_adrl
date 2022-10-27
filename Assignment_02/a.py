import torch

# x = torch.tensor([[1,1,1],[2,2,2]]).permute(1,0)
# print(x.size())

# w = torch.tensor([[3,3]])
# print(w.size())

# y = torch.mul(w,x)
# print(y)
# print(y.size())

x = torch.tensor([[1,2,3,4,2,6,7]])
y = (x == 22).nonzero(as_tuple=False)
print(y)
print(y.numel())
# print(x.size())

# z = torch.randn(50,100)
# print(z[y[:,1]].size())

# aa = torch.randn((128,1200))
# z = 0
# k = aa + z
# print(k.size())