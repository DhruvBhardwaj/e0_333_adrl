import torch
import numpy as np


def compute_entropy(img):
    c,h,w = img.size()
    for bin_lower in range(0,256,1):
        print(bin_lower,bin_lower+1)

if __name__ == '__main__':
    x = torch.randn(2,128,128)
    compute_entropy(x)