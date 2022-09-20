import torch
torch.manual_seed(42)

from torch.nn import Conv2d, ConvTranspose2d, Linear, Embedding
from torch.nn import MaxPool2d, BatchNorm2d
from torch.nn import LeakyReLU, Tanh, ReLU, Sigmoid
from torch.nn import Module
from torch.nn import MSELoss
from torch import flatten

class Encoder(Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_dim = self.latent_dim

        self.conv1 = Conv2d(in_channels=self.in_channels, out_channels=32,kernel_size=3, stride=2, padding=1)
        self.bnorm1 = BatchNorm2d(32)
        self.relu1 = LeakyReLU()        
        
        self.conv2 = Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=2, padding=1)
        self.bnorm2 = BatchNorm2d(64)
        self.relu2 = LeakyReLU()

        self.conv3 = Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=2, padding=1)
        self.bnorm3 = BatchNorm2d(128)
        self.relu3 = LeakyReLU()

        self.conv4 = Conv2d(in_channels=128, out_channels=256,kernel_size=3, stride=2, padding=1)
        self.bnorm4 = BatchNorm2d(256)
        self.relu4 = LeakyReLU()
        
        #self.fc1 = Linear(in_features=256*4, out_features=256)	        

        self.fcMu = Linear(in_features=256*4*4, out_features=self.output_dim)
        self.fcCov = Linear(in_features=256*4*4, out_features=self.output_dim)
        self.relu5 = ReLU()

        return

    def forward(self, x):

        x = self.bnorm1(self.relu1((self.conv1(x))))
        x = self.bnorm2(self.relu2((self.conv2(x))))
        x = self.bnorm3(self.relu3((self.conv3(x))))
        x = self.bnorm4(self.relu4((self.conv4(x))))
        
        x = flatten(x, start_dim=1)                
        out_mu = self.fcMu(x)
        out_cov = self.relu5(self.fcCov(x))

        #out_cov = self.fcCov(x) # log variance
        out = torch.cat((out_mu, out_cov),dim=1)
        return out

class Decoder(Module):
    def __init__(self, latent_dim, output_dim, output_channels):
        super(Decoder, self).__init__()

        self.input_dim = latent_dim
        self.output_dim = output_dim
        self.output_channels = output_channels

        self.fc1 = Linear(self.input_dim, out_features=256*4*4)	        
        
        self.convT1 = ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bnorm1 = BatchNorm2d(128)
        self.relu1 = LeakyReLU()

        self.convT2 = ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bnorm2 = BatchNorm2d(64)
        self.relu2 = LeakyReLU()

        self.convT3 = ConvTranspose2d(in_channels=64, out_channels=32,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bnorm3 = BatchNorm2d(32)
        self.relu3 = LeakyReLU()

        self.convT31 = ConvTranspose2d(in_channels=32, out_channels=32,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bnorm31 = BatchNorm2d(32)
        self.relu31 = LeakyReLU()
        
        self.convT4 = ConvTranspose2d(in_channels=32, out_channels=self.output_channels,kernel_size=3, padding=1)
        #self.bnorm4 = BatchNorm2d(3)
        self.tanh = Tanh()            

        return
    
    def forward(self,x):
        
        x = self.fc1(x)        
        
        x = torch.reshape(x,(-1,256,4,4))
        
        x = self.bnorm1(self.relu1(self.convT1(x)))
        
        x = self.bnorm2(self.relu2(self.convT2(x)))
        
        x = self.bnorm3(self.relu3(self.convT3(x)))
        
        x = self.bnorm31(self.relu31(self.convT31(x)))
        
        out = self.tanh(self.convT4(x))     
         
        return 0.5 + (0.5*out)

# Define VAE class
class Varn_AE(Module):
    def __init__(self, input_dim, latent_dim, in_channels, var_norm):
        super().__init__()

        self.eps = 0.000001
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.var_norm = var_norm

        self.mse_loss = torch.nn.MSELoss(reduction='sum')        
        self.kld_loss = self.kld_gaussian

        self.mse_loss_value = 0.0
        self.kl_loss_value = 0.0

        self.mu = torch.zeros(self.latent_dim)
        self.dCov = torch.zeros(self.latent_dim)
        
        self.encoder = Encoder(self.in_channels,self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim, self.in_channels)
        
        self.encoder.float()
        self.decoder.float()

        print()
        print('-'*59)
        print('ENCODER MODEL')
        print('-'*59)
        print(self.encoder)

        print('-'*59)
        print('DECODER MODEL')
        print('-'*59)
        print(self.decoder)
        print()
        return

    def train(self):
        super().train()
        if(self.encoder != None):
            self.encoder.train()
        if(self.decoder != None):
            self.decoder.train()
        return  

    def forward(self,x):        
        
        ## ENCODE ##
        x = self.encoder(x)

        ## SAMPLE ##
        self.mu = x[:,:self.latent_dim]
        self.dCov = x[:,self.latent_dim:] + self.eps
        
        eps = torch.randn_like(self.dCov)                
        z = self.mu + (eps*self.dCov)

        ## DECODE ##
        x_hat=self.decoder(z)

        return x_hat
    
    def kld_gaussian(self):   
        mu_sq = torch.square(self.mu)
        loss_j = (1.0 + torch.log(self.dCov)) - mu_sq - self.dCov
        kld_loss = -1*0.5*torch.sum(loss_j)
        return kld_loss

    def criterion(self, x, x_hat):            
                
        x_hat = torch.reshape(x_hat, (x.size(0),-1))        
        
        x = torch.reshape(x,(x.size(0),-1,))        
        
        self.mse_loss_value = self.mse_loss(x,x_hat)

        self.mse_loss_value = self.mse_loss_value/(2*self.var_norm)
        self.kl_loss_value = self.kld_loss()        
        
        return self.kl_loss_value + self.mse_loss_value, self.mse_loss_value, self.kl_loss_value        
    
    def sample(self,num_samples=100, curr_device="cpu"):
        self.decoder.eval()
        z = torch.randn((num_samples,self.latent_dim),device=curr_device)
        samples = self.decoder(z)
        #print(samples.size())
        self.decoder.train()
        return samples

class VQ(Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings        

        self.mse_loss = torch.nn.MSELoss(reduction='sum')                
                
        self.embedding_loss_value = 0.0
        self.commitment_loss_value = 0.0            
        
        self.dictionary = Embedding(self.num_embeddings, self.embedding_dim)
        self.dictionary.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
                
        return        

    def findNearest(self,x):
        # x is BATCH x EMBED_DIM
        # xref is NUM_EMBED X EMBED_DIM

        xref = self.dictionary.weight

        x_norm2 = torch.linalg.norm(x, dim=1,keepdim=True)**2 #[BATCH x 1]
        x_norm2 = x_norm2.expand(-1,self.num_embeddings) #[BATCH x NUM_EMBED]

        xref_norm2 = torch.linalg.norm(xref, dim=1,keepdim=True)**2 #[NUM_EMBED x 1]
        xref_norm2 = xref_norm2.expand(-1, x.size(0)) #[NUM_EMBED x BATCH]
        xref_norm2 = torch.transpose(xref_norm2, 0, 1) #[BATCH X NUM_EMBED]        
        
        dist = x_norm2 + xref_norm2 - 2*torch.matmul(x, torch.transpose(xref, 0 , 1))        

        nearest_idxs =  torch.argmin(dist, dim=1).unsqueeze(1)
        
        return nearest_idxs

    def forward(self,z_e):        
        
        ## Find nearest embeddings      
        nearest_idxs = self.findNearest(z_e)
        
        ## DECODE ##
        z_q = self.dictionary.weight[nearest_idxs,:].squeeze()   
        
        self.embedding_loss_value = self.mse_loss(z_e.detach(),z_q)
        self.commitment_loss_value = self.mse_loss(z_e,z_q.detach())

        self.vq_loss = self.embedding_loss_value + self.commitment_loss_value
        
        z_q = z_e + (z_q - z_e).detach()

        return z_q, self.vq_loss    

class VQ_Varn_AE(Module):
    def __init__(self, input_dim, embedding_dim, in_channels, var_norm, num_embeddings, beta):
        super().__init__()
                
        self.input_dim = input_dim        
        self.in_channels = in_channels
        self.var_norm = var_norm
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.mse_loss = torch.nn.MSELoss(reduction='sum')                
        
        self.mse_loss_value = 0.0        
                
        #print(self.dictionary.weight.size())
        self.encoder = Encoder(self.in_channels,int(0.5*self.embedding_dim))
        self.vq = VQ(self.embedding_dim, self.num_embeddings)
        self.decoder = Decoder(self.embedding_dim, self.input_dim, self.in_channels)
        
        self.encoder.float()
        self.vq.float()
        self.decoder.float()

        print()
        print('-'*59)
        print('ENCODER MODEL')
        print('-'*59)
        print(self.encoder)

        print('-'*59)
        print('VQ LAYER')
        print('-'*59)
        print(self.vq)

        print('-'*59)
        print('DECODER MODEL')
        print('-'*59)
        print(self.decoder)
        print()

        return
    
    def train(self):
        super().train()
        if(self.encoder != None):
            self.encoder.train()
        if(self.vq != None):
            self.vq.train()
        if(self.decoder != None):
            self.decoder.train()
        return     

    def forward(self,x):        
        ## ENCODE ##
        z_e = self.encoder(x)
        
        ## Find nearest embeddings      
        z_q, vq_loss = self.vq(z_e)
        
        ## DECODE ##                
        x_hat = self.decoder(z_q)        

        return x_hat, vq_loss
    
    def criterion(self, x, x_hat, vq_loss):            
                
        x_hat = torch.reshape(x_hat, (x.size(0),-1))                
        x = torch.reshape(x,(x.size(0),-1,))        
        
        self.mse_loss_value = self.mse_loss(x,x_hat)
        self.mse_loss_value = self.mse_loss_value/(2*self.var_norm)        
        
        total_loss = self.mse_loss_value + vq_loss

        return total_loss, self.mse_loss_value
    
    def sample(self, x_in=None, num_samples=100, curr_device="cpu"):
        self.decoder.eval()
        self.vq.eval()
        self.encoder.eval()

        if(x_in == None):
            return None
        else:
            z_e = self.encoder(x_in)              
            z_q, _ = self.vq(z_e)                
            samples = self.decoder(z_q)                    
        
        self.decoder.train()
        self.vq.train()
        self.encoder.train()
        
        return samples

