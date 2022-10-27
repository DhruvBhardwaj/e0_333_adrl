import torch
from torch import nn
import numpy as np
from torchvision.transforms import CenterCrop
from config_1a_celeba import cfg

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ConvBlock,self).__init__()
        
        layers=[]

        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size=3))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.net(x)

class uNet(nn.Module):
    def __init__(self,cfg):
        super(uNet,self).__init__()

        self.encoder_cfg = cfg['encoder']
        self.decoder_cfg = cfg['decoder']

        layers=[]

        for i in range(len(self.encoder_cfg['layers'])-1):
            layer = self.encoder_cfg['layers'][i]
            for j in range(len(layer)-1):
                layers.append(ConvBlock(layer[j],layer[j+1]))
            layers.append(nn.MaxPool2d(2))
        
        layer = self.encoder_cfg['layers'][-1]
        for j in range(len(layer)-1):
            layers.append(ConvBlock(layer[j],layer[j+1]))

        self.encodernet = nn.ModuleList(layers)
        print(self.encodernet)

        layers=[]

        for i in range(len(self.decoder_cfg['layers'])):
            layer = self.decoder_cfg['layers'][i]
            layers.append(nn.ConvTranspose2d(layer[0],int(0.5*layer[0]),kernel_size=2,stride=2))
            for j in range(len(layer)-1):
                layers.append(ConvBlock(layer[j],layer[j+1]))            
        
        layers.append(nn.Conv2d(self.decoder_cfg['layers'][-1][-1],self.decoder_cfg['out_ch'],kernel_size=1))

        self.decodernet = nn.ModuleList(layers)        
        self.upconv1 = nn.ConvTranspose2d(3,3,kernel_size=3,stride=3, padding=4)
        self.tanh = nn.Tanh()

        print('-'*59)
        print('---uNET---')        
        print('---ENCODER---')
        print(self.encodernet)
        print('-'*59)        
        print('---DECODER---')
        print(self.decodernet)
        print(self.upconv1)
        print(self.tanh)
        print('-'*59)

    def forward(self,x):
        
        enc_out=[]
        for i in range(len(self.encodernet)):
            layer = self.encodernet[i]
            if isinstance(layer, nn.MaxPool2d):        
                enc_out.append(x) #save input of maxpool
            x = layer(x)
        
        enc_out.reverse()
        
        x = self.decodernet[0](x)
        h,w = x.size(2),x.size(3)                
        x = torch.cat([x,CenterCrop([h, w])(enc_out[0])],dim=1)

        k=1
        for i in range(1,len(self.decodernet)):
            layer = self.decodernet[i]
            x = layer(x)

            if isinstance(layer, nn.ConvTranspose2d):
                h,w = x.size(2),x.size(3)                
                x = torch.cat([x,CenterCrop([h, w])(enc_out[k])],dim=1)
                k += 1

        x = self.tanh(self.upconv1(x))
        x = 0.5*(x+1)
        return x

class DiffusionNet(nn.Module):
    def __init__(self,cfg, device):
        super(DiffusionNet,self).__init__()

        self.cfg = cfg['diffusion']        
        self.net = uNet(cfg)
        self.device = device
        self.alpha_t, self.alphabar_t = self.getLinearSchedule()        
        self.P = self.getPosnEncode(self.cfg['T'],self.cfg['d'])
        self.bce_loss = nn.BCELoss(reduction='sum')
        

    def getLinearSchedule(self):
        beta1 = self.cfg['BETA1']
        betaT = self.cfg['BETAT']
        T = self.cfg['T']

        m = (betaT-beta1)/(T-1)
        c = beta1 - m
        
        alpha_t = torch.tensor([(1-(m*t+c)) for t in range(1,T+1)]).to(self.device)
        alphabar_t = torch.cumprod(alpha_t, dim=0).to(self.device)
        
        return alpha_t, alphabar_t
    
    def getPosnEncode(self,T,d):        
        T = int(T)
        d = int(d)
        P = np.zeros((T,d))
        n_2d = (10000)**(2/d)
        for k in range(int(T/2)):
            t = k + 1
            for i in range(int(d/2)):                
                P[k][2*i] = np.sin(t/(n_2d**i))
                P[k][1+2*i]= np.cos(t/(n_2d**i))

        return torch.from_numpy(P)

    def getNoisySample(self,x):
        t = self.t
        b,c,h,w = x.size()
        x = torch.reshape(x,(b,c*h*w)).permute(1,0)
        
        alphabar_t = self.alphabar_t[t]

        e0 = torch.randn_like(x).float().to(self.device)
        
        e = torch.mul(((1-alphabar_t)**0.5),e0)
        e = e + torch.mul((alphabar_t**0.5),x)

        e = torch.reshape(e.permute(1,0),(b,c,h,w)).float()
        return e, e0

    def forward(self,x):
        self.t = torch.randint(low=0,high=self.cfg['T']-1,size=(1,x.size(0)))
        
        e,e0 = self.getNoisySample(x)
        e = e + (self.P[self.t].squeeze(0)).reshape(e.size(0),e.size(1),e.size(2),e.size(3)).float().to(self.device)
        
        e = self.net(e)        
        return e, e0

    def criterion(self,x,e,e0):
        
        t = self.t
        t_is1 = (t == 0).nonzero(as_tuple=False)
        t_isNot1 = (t != 0).nonzero(as_tuple=False)
        
        e = torch.flatten(e,start_dim=1)
        e0 = e0.permute(1,0)
        
        diff_e = e[t_isNot1[:,1]]-e0[t_isNot1[:,1]]
        diff_norm = torch.linalg.vector_norm(diff_e,dim=1,keepdim=True)        
        loss1 = torch.sum(diff_norm**2)
        
        # For eq (13) only
        # alphabar_t = self.alphabar_t[t]
        # alpha_t = self.alpha_t[t]
        # const = ((1-alpha_t)/(2*(alpha_t)*(1-alphabar_t))).permute(1,0)
        #loss = torch.sum(const*diff_norm) 

        if(t_is1.numel()>0):            
            x = torch.flatten(x,start_dim=1)
            loss2 = self.bce_loss(e[t_is1[:,1]],x[t_is1[:,1]])            
        else:
            loss2 = torch.tensor([0]).to(self.device)            

        loss = loss1 + loss2
        return loss, loss1, loss2

if __name__ == '__main__':

    # m = uNet(cfg)
    x = torch.randn(5,3,64,64)
    x = -torch.min(x) + (x/(torch.max(x)-torch.min(x)))
    # y = m(x)
    
    d = DiffusionNet(cfg,'cpu')
    e, e0 = d(x)
    #loss = d.criterion(x,e,e0)
    #print(loss)
    
    