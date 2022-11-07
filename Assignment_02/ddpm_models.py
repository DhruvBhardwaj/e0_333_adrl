import math
from inspect import isfunction
from functools import partial

from tqdm.auto import tqdm
import utils as util
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms
transform = transforms.Compose([        
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        ])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,        
        convnext_mult=2,
        encoder_only=False
    ):
        super().__init__()
        self.encoder_only=encoder_only
        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
                
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        if self.encoder_only is False:
            print('decoder')
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (num_resolutions - 1)

                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            Upsample(dim_in) if not is_last else nn.Identity(),
                        ]
                    )
                )

            out_dim = default(out_dim, channels)
            self.final_conv = nn.Sequential(
                block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
            )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        if self.encoder_only is False:            
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)
            
            x = self.final_conv(x)

        return x

class DiffusionClassifier(nn.Module):
    def __init__(self,cfg,num_classes, device):
        super(DiffusionClassifier,self).__init__()
        self.cfg=cfg
        self.betas = linear_beta_schedule(timesteps=self.cfg['diffusion']['T'])        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.num_classes = num_classes
        self.net=Unet(dim=cfg['ddpm']['image_size'], channels=cfg['ddpm']['channels'],dim_mults=(1, 2, 4,),encoder_only=True)
        
        self.dense1 = nn.Linear(256*16*16,1024)
        self.dense2 = nn.Linear(1024,self.num_classes)

        self.device = device
        print(self.net)

    def forward(self,x, t):
        self.t = t
        e,_ = self.q_sample(x)
        x = self.net(e,self.t)
        x = torch.flatten(x,start_dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def score_fn(self,x,t):
        with torch.enable_grad():            
            x_in = x.detach().requires_grad_(True)
            x_in = transform(x_in)
            out = self.forward(x_in, t)
            log_probs = F.log_softmax(out, dim=1)            
            selected = torch.diagonal(log_probs)
            
            score = torch.autograd.grad(selected.sum(), x_in)
            
            return score[0]

    def q_sample(self,x_start, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, self.t, x_start.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, self.t, x_start.shape
        ).to(self.device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

class DiffusionNet(nn.Module):
    def __init__(self,cfg, device):
        super(DiffusionNet,self).__init__()

        self.cfg = cfg['diffusion']        
        self.betas = linear_beta_schedule(timesteps=self.cfg['T'])
        self.device = device
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.net=Unet(dim=cfg['ddpm']['image_size'], channels=cfg['ddpm']['channels'],dim_mults=(1, 2, 4,))

        print(self.net)        
            
                
    def q_sample(self,x_start, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, self.t, x_start.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, self.t, x_start.shape
        ).to(self.device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def p_sample(self,x, t, t_index, classifier=None):
        betas_t = extract(self.betas, t, x.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        ).to(self.device)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape).to(self.device)

        if(classifier is not None):
            classifier.eval()
            classifier_score = sqrt_one_minus_alphas_cumprod_t*10*classifier.score_fn(x,t)
            model_mean = sqrt_recip_alphas_t * (
            x - betas_t * (self.net(x, t)-classifier_score.to(self.device)) / sqrt_one_minus_alphas_cumprod_t
            )
        else:
            model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.net(x, t) / sqrt_one_minus_alphas_cumprod_t
            )
        

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape).to(self.device)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def p_sample_loop(self, classifier, shape):

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.cfg['T'])), desc='sampling loop time step', total=self.cfg['T']):
            img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), i, classifier)
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, classifier=None):
        return self.p_sample_loop(classifier, shape=(batch_size, channels, image_size, image_size))

        
    def forward(self,x):
        self.t = torch.randint(low=0,high=self.cfg['T']-1,size=(x.size(0),),device=self.device).long()
        
        e,e0 = self.q_sample(x)
        #print(e.size(),e0.size())
        e = self.net(e, self.t)  
        #print(e.size())      
        return e, e0

    def criterion(self,e,e0):
           
        loss = F.mse_loss(e,e0, reduction='sum')
        return loss

class EBM(nn.Module):
    def __init__(self,cfg, device):
        super(EBM,self).__init__()
        self.cfg=cfg        
        self.eps = cfg['ebm']['sample_eps']
        self.T = cfg['ebm']['num_steps']
        self.eps2 = self.eps**2        
        self.net=Unet(dim=cfg['ddpm']['image_size'], channels=cfg['ddpm']['channels'],dim_mults=(1, 2, 4,),encoder_only=True,with_time_emb=False)
        
        self.dense1 = nn.Linear(256*16*16,1024)
        self.dense2 = nn.Linear(1024,512)
        self.dense3 = nn.Linear(512,1)

        self.device = device
        print(self)

    def forward(self,x):
                
        x = self.net(x,None)
        x = torch.flatten(x,start_dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def criterion(self, es, e):        
        return torch.mean(es) - torch.mean(e) + 0.001*(torch.mean(es**2 + e**2))

    def sample(self,x):
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        x.requires_grad = True
        noise = torch.randn_like(x, device=self.device)
        for t in range(0,self.T):
            noise.normal_(0,0.001)
            x.data.add_(noise.data)
            x.data.clamp_(-1.0, 1.0)
            
            energy = -1*self.forward(x)
            energy.sum().backward()    
                        
            x.data.add_(-0.5*self.eps2*x.grad.data.clamp_(-0.03, 0.03))
            #x.data.add_(-10*x.grad.data.clamp_(-0.03, 0.03))
            x.grad.detach_()
            x.grad.zero_()
            x.data.clamp_(-1.0, 1.0)             
        torch.set_grad_enabled(had_gradients_enabled)
        return x

if __name__ == '__main__':
    from config_1a_celeba import cfg
    #from config_1a_bitmojis import cfg
    
    d = DiffusionClassifier(cfg,10,'cpu')
    x = torch.randn(2,3,64,64)
    y = d(x)
    print(y.size())
    print(y)
    
    # x = d.sample(cfg['ddpm']['image_size'],100,cfg['ddpm']['channels'])            
    # print(len(x))
    # print(x[0].size())
    # util.save_image_to_file(999,0.5*(x[-1]+1),cfg['training']['save_path'],'bitm')