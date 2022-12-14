import torch
from torch import nn
from torchvision.transforms import CenterCrop
import numpy as np
import torch.nn.functional as F

def pEncode(t, d):
    pEmb = torch.zeros((t.size(1),d))
    logn_d = -1.0*(torch.log(torch.tensor([10000])))/d
    exp_d = torch.exp(torch.arange(0,d,2)*logn_d)
    
    pEmb[:,0::2] = torch.sin(torch.outer((t+1).squeeze(0),exp_d))
    pEmb[:,1::2] = torch.cos(torch.outer((t+1).squeeze(0),exp_d))
    
    return pEmb

def getPosnEncode(T,d):        
        
    P = np.zeros((T,d))
    n_2d = (10000)**(2/d)
    for k in range(int(T)):
        t = k + 1
        for i in range(int(d/2)):                
            P[k][2*i] = np.sin(t/(n_2d**i))
            P[k][1+2*i]= np.cos(t/(n_2d**i))

    return torch.from_numpy(P)

class posnEncoder(nn.Module):
    def __init__(self, device='cpu', d=None):
        super(posnEncoder,self).__init__()
        self.device=device
        self.d = d
        if(self.d is not None):
            self.logn_d = -1.0*(torch.log(torch.tensor([10000])))/self.d
            self.exp_d = torch.exp(torch.arange(0,self.d,2)*self.logn_d)
    
    def forward(self,x,t=None):

        if t is None:
            return x
        # t is 1xb, x = bxcxhxw
        
        b,c,h,w = x.size()
        if(self.d is None):
            self.d = c*h*w
            self.logn_d = -1.0*(torch.log(torch.tensor([10000])))/self.d
            self.exp_d = torch.exp(torch.arange(0,self.d,2)*self.logn_d)
        
        pEmb = torch.zeros((b,self.d)).to(self.device)
        x = x.reshape(b,self.d)
        ct = torch.outer((t+1).squeeze(0),self.exp_d).to(self.device)


        pEmb[:,0::2] += ct.sin()
        pEmb[:,1::2] += ct.cos()

        x = x + pEmb
        del pEmb
        x = x.reshape(b,c,h,w)
        return x

class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(convBlock,self).__init__()
        self.device = device
        layers =[]

        layers.append(nn.Conv2d(in_channels,out_channels[0],3))
        layers.append(nn.Dropout(0.1))
        layers.append(nn.BatchNorm2d(out_channels[0]))
        #layers.append(nn.GroupNorm(4, out_channels[0]))        
        layers.append(nn.ReLU())
        #layers.append(posnEncoder(self.device))
        layers.append(nn.Conv2d(out_channels[0],out_channels[1],3))
        layers.append(nn.Dropout(0.1))
        layers.append(nn.BatchNorm2d(out_channels[1]))
        #layers.append(nn.GroupNorm(4, out_channels[1]))
        layers.append(nn.ReLU())    
        layers.append(posnEncoder(self.device))    

        self.net = nn.ModuleList(layers)#nn.Sequential(*layers)        

    def forward(self, x, t=None):        
        for i, l in enumerate(self.net):
            #x=l(x)
            if isinstance(l, posnEncoder) and (t is not None):
                x = l(x,t)                                    
                #d = x.size(1)*x.size(2)*x.size(3)
                #x = x + pEncode(t,d).reshape(x.size(0),x.size(1),x.size(2),x.size(3)).to(self.device)               
            else:
                x = l(x)
        
        return x

class uNet(nn.Module):
    def __init__(self, cfg, device):
        super(uNet,self).__init__()        
        self.device = device
        enc_channels=cfg['encoder']['layers'] 
        dec_channels=cfg['decoder']['layers']

        ### Unet Encoder ###
        layers = []

        layers.append(convBlock(enc_channels[0][0],enc_channels[0][1:],self.device))

        for i in range(1,len(enc_channels)):
            layers.append(nn.MaxPool2d((2,2))) 
            layers.append(convBlock(enc_channels[i][0],enc_channels[i][1:],self.device))

        self.encoderNet = nn.ModuleList(layers)
        
        ### Unet Decoder ###
        layers = []
        for i in range(0,len(dec_channels)):
            layers.append(nn.ConvTranspose2d(dec_channels[i][0],dec_channels[i][1],2,2))
            layers.append(convBlock(2*dec_channels[i][1],dec_channels[i][1:],self.device))            

        self.decoderNet = nn.ModuleList(layers)

        ### Unet Output layers ###
        layers = []
        self.upconv = nn.ConvTranspose2d(dec_channels[-1][-1],int(0.5*dec_channels[-1][-1]),3,3,padding=4)
        self.conv1 = nn.Conv2d(int(0.5*dec_channels[-1][-1]),enc_channels[0][0],1)
        self.tanh = nn.Tanh()

        print('-'*59)
        print('ENCODER')
        print('-'*59)
        print(self.encoderNet)
        print('-'*59)
        print('-'*59)
        print('DECODER')
        print('-'*59)
        print(self.decoderNet)        
        print('-'*59)
        print(self.upconv)
        print('-'*59)
        print(self.conv1)
        print('-'*59)
        print(self.tanh)
        print('-'*59)

    def forward(self,x, t):
        H,W = x.size(2), x.size(3)
        
        enc_out=[]
        for i, l in enumerate(self.encoderNet):
            if isinstance(l,convBlock):                
                x = l(x,t)
            else:
                x = l(x)
            enc_out.append(x)

        N = len(enc_out)
        for i, l in enumerate(self.decoderNet):
            if(i%2==0):
                x = l(x)
            else:
                enc_map = CenterCrop([x.size(2), x.size(3)])(enc_out[N-2-i])         
                x = l(torch.cat([x, enc_map], axis=1), t)                
            
        x = self.tanh(self.conv1(self.upconv(x)))
        #x = 0.5*(x+1)   
        return x

class DiffusionNet(nn.Module):
    def __init__(self,cfg, device):
        super(DiffusionNet,self).__init__()

        self.cfg = cfg['diffusion']        
        self.device = device
        self.net = uNet(cfg, self.device)        
        self.alpha_t, self.alphabar_t = self.getLinearSchedule()  
        self.beta_t = 1 - self.alpha_t      
        self.sample_const = (self.beta_t)/((1-self.alphabar_t)**0.5)  
        self.sample_const2 = (1/(self.alpha_t**0.5))     
        self.posnEncode = posnEncoder(self.device) 
        #self.bce_loss = nn.BCELoss(reduction='sum')
        
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
        
    def getLinearSchedule(self):
        beta1 = self.cfg['BETA1']
        betaT = self.cfg['BETAT']
        T = self.cfg['T']

        m = (betaT-beta1)/(T-1)
        c = beta1 - m
        
        alpha_t = torch.tensor([(1-(m*t+c)) for t in range(1,T+1)]).to(self.device)
        alphabar_t = torch.cumprod(alpha_t, dim=0).to(self.device)
        
        return alpha_t, alphabar_t

    def forward(self,x):
        self.t = torch.randint(low=0,high=self.cfg['T']-1,size=(1,x.size(0)))
        
        e,e0 = self.getNoisySample(x)        
        
        #e = e + pEncode(self.t,e.size(1)*e.size(2)*e.size(3)).reshape(e.size(0),e.size(1),e.size(2),e.size(3)).to(self.device)
        e = self.posnEncode(e,self.t)
        e = self.net(e, self.t)        
        return e, e0

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

    def criterion(self,e,e0):
            
            e = torch.flatten(e,start_dim=1)
            e0 = e0.permute(1,0)
            
            #diff_e = e-e0
            #diff_norm = torch.linalg.norm(diff_e,dim=1,keepdim=True)        
            #loss = torch.sum(diff_norm**2)
            loss = F.mse_loss(e,e0, reduction='sum')
            return loss

    def sample(self,N=10, end_T=1):        
        samples=[]        

        x = torch.randn((N,3,64,64)).to(self.device)
        for t in range(self.cfg['T']-1,end_T-1,-1):
            print(t,end = " ")
            if(t>1):
                z = ((self.beta_t[t]**0.5))*(torch.randn((N,3,64,64)).to(self.device))
            else:
                z = 0

            e=self.net(x,t*torch.ones((1,N)))
            
            x = x-(self.sample_const[t]*e)
            x = self.sample_const2[t]*x
            #x = x + z                                            
        return x

if __name__ == '__main__':
    #from config_1a_celeba import cfg
    from config_1a_bitmojis import cfg

    d = DiffusionNet(cfg,'cpu')
    d.train(True)
   
    x = torch.randn(2,3,128,128)
    y,_ = d(x)
    print(y.size())
    
    # x = torch.zeros((2,1,2,5))
    
    # t = torch.randint(low=0,high=10-1,size=(1,2))    
    # print(t)
    # pemb = pEncode(t,10)
    # print(torch.max(pemb),torch.min(pemb))
    # print(pemb)
    # P = getPosnEncode(10,10)
    # print('----')
    # print(P[t])
    # print('----')
    # pm = posnEncoder('cpu')
    # x = pm(x,t)
    # print(x)

    # print(pemb.reshape(5,3,256,256).numpy().shape)
    # import matplotlib.pyplot as plt
    # img = pemb.reshape(5,3,256,256).numpy()[1].T
    # img = 0.5*(img+1)
    # plt.imsave('temp1.png',img, cmap='gray')
