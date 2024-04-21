from turtle import forward
import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
import math
from util.util import get_coord
import numpy as np
from einops import rearrange
import numbers

class NIRRestormerModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(NIRRestormerModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['RT_MSE', 'RT_MSE2', 'RT_MSE4', 'Total']
        self.visual_names = ['data_ir', 'data_gt_noise', 'data_out', 'data_gt']
        self.model_names = ['RT'] 
        self.optimizer_names = ['RT_optimizer_%s' % opt.optimizer]

        unet = NIRRestormer(opt)
        self.netFB = N.init_net(unet, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:		
            self.optimizer_RT = optim.Adam(self.netRT.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)
        
            self.optimizers = [self.optimizer_RT]

            self.criterionMSE = N.init_net(MSELoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input, epoch):
        self.data_gt = input['gt_img'].to(self.device)
        self.data_ir = input['ir_img'].to(self.device)
        self.data_gt_noise = input['noise_img'].to(self.device)
        self.image_paths = input['fname']
        self.epoch = epoch

    def forward(self):
        if self.isTrain:
            h ,w = self.data_ir.shape[-2:]
            if self.epoch != -1:
                self.data_out, self.out2, self.out4 = self.netFB(self.data_gt_noise, self.data_ir)
                self.data_out = self.data_out[..., :h, :w]
            else:
                self.data_out = torch.zeros_like(self.data_gt_noise)
                self.data_out_0, self.out2, self.out4 = self.netFB(self.data_gt_noise[..., :h, :w//2], self.data_ir[..., :w//2])
                self.data_out_1, self.out2, self.out4 = self.netFB(self.data_gt_noise[..., :h, w//2 : w], self.data_ir[..., w//2 : w])
                self.data_out[..., : w//2]  =  self.data_out_0
                self.data_out[..., w//2 : w]  =  self.data_out_1
        else:
            h ,w = self.data_ir.shape[-2:]
            self.data_out = torch.zeros_like(self.data_gt_noise)
            self.data_out_0, self.out2, self.out4 = self.netFB(self.data_gt_noise[..., :h, :w//2], self.data_ir[..., :w//2])#[...,:h//2,:w//2]
            self.data_out_1, self.out2, self.out4 = self.netFB(self.data_gt_noise[..., :h, w//2 : w], self.data_ir[..., w//2 : w])#[...,:h//2,:w//2]
            self.data_out[..., : w//2]  =  self.data_out_0
            self.data_out[..., w//2 : w]  =  self.data_out_1



    def backward(self):
        _, _, H, W = self.data_gt.shape
        data_gt2 = F.interpolate(self.data_gt, scale_factor=0.5, mode='bilinear')[:, :, :H//2, :W//2]
        data_gt4 = F.interpolate(self.data_gt, scale_factor=0.25, mode='bilinear')[:, :, :H//4, :W//4]
        self.loss_RT_MSE = self.criterionMSE(self.data_out, self.data_gt).mean()
        self.loss_RT_MSE2 = self.criterionMSE(self.out2, data_gt2).mean()
        self.loss_RT_MSE4 = self.criterionMSE(self.out4, data_gt4).mean()
        self.loss_Total = self.loss_RT_MSE + self.loss_RT_MSE2 + self.loss_RT_MSE4
        self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_RT.zero_grad()
        self.backward()
        self.optimizer_RT.step()


class GMM(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv_rgb = N.conv(c, c, kernel_size=1, padding=0, mode='C')
        self.conv_nir = N.conv(c, c, kernel_size=1, padding=0, mode='C')
        self.softmax=nn.Softmax(dim=1)
        self.pool= nn.Sequential(N.conv(c, c, kernel_size=1, padding=0, mode='C'),
            LayerNorm2d(c),
            nn.PReLU(),
            N.conv(c, c, kernel_size=1, padding=0, mode='C'),
            LayerNorm2d(c),
            nn.PReLU())
        self.fc_ab = nn.Sequential(
            N.conv(c,c*2,1,padding=0,bias=False, mode='C'))

    def forward(self, rgb, nir):
        feat_1 = self.conv_rgb(rgb)
        feat_2 = self.conv_nir(nir)
        feat_sum = feat_1 + feat_2
        s = self.pool(feat_sum)
        z = s
        ab = self.fc_ab(z)
        B, C, H, W = ab.shape
        ab=ab.view(B,2, C//2,H,W)
        ab=self.softmax(ab)
        a = ab[:,0,...]
        b = ab[:,1,...]
        feat_1 = a * feat_1
        feat_2 = b * feat_2
        return feat_1, feat_2
    
class LMM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = N.conv(2*c, c, kernel_size=1, padding=0, mode='C')
        self.pool_5= nn.Sequential(N.conv(c, c, kernel_size=5, padding=2, mode='C', groups=c),
            LayerNorm2d(c),
            nn.PReLU(),
            N.conv(c, c, kernel_size=5, padding=2, mode='C', groups=c),
            LayerNorm2d(c),
            nn.PReLU())
        self.fc_ab = nn.Sequential(
            N.conv(c,c*2,1,padding=0,bias=False, mode='C'))
        self.softmax=nn.Softmax(dim=1)

    def forward(self, rgb, nir):
        feat_cat = torch.cat((rgb, nir), dim=1)
        feat = self.conv(feat_cat)
        s = self.pool_5(feat)
        z = s
        ab = self.fc_ab(z)
        B, C, H, W = ab.shape
        ab=ab.view(B,2, C//2,H,W)
        ab=self.softmax(ab)
        a = ab[:,0,...]
        b = ab[:,1,...]
        feat_1 = a * rgb
        feat_2 = b * nir
        return feat_1, feat_2
    
class NIRRestormer(nn.Module):
    def __init__(self, opt):
        super(NIRRestormer, self).__init__()
        self.opt = opt
        ch_1 = 48
        LayerNorm_type = 'WithBias'
        ffn_expansion_factor = 2.66
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        transblocks = [4,6,6,8]
        bias = False
        self.GMMs = nn.ModuleList()
        self.LMMs = nn.ModuleList()

        self.patch_embed = N.conv(3, ch_1, mode='C')
        self.patch_embe_n = N.conv(1, ch_1, mode='C')

        self.encoder_1 = nn.Sequential(*[TransformerBlock(dim=ch_1, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[0])])
        self.GMMs.append(GMM(ch_1))
        self.LMMs.append(LMM(ch_1))
        self.down1_2 = Downsample(ch_1)
        self.encoder_2 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[1])])
        self.GMMs.append(GMM(ch_1*2**1))
        self.LMMs.append(LMM(ch_1*2**1))
        self.down2_3 = Downsample(int(ch_1*2**1)) ## From Level 2 to Level 3
        self.encoder_3 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[2])])
        self.GMMs.append(GMM(ch_1*2**2))
        self.LMMs.append(LMM(ch_1*2**2))

        #### for nir #####
        self.encoder_n1 = nn.Sequential(*[TransformerBlock(dim=ch_1, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[0])])
        self.down1_n2 = Downsample(ch_1)
        self.encoder_n2 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[1])])
        self.down2_n3 = Downsample(int(ch_1*2**1)) ## From Level 2 to Level 3
        self.encoder_n3 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[2])])


        self.down3_4 = Downsample(int(ch_1*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[3])])
        self.up4_3 = Upsample(int(ch_1*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_3 = nn.Conv2d(int(ch_1*2**3), int(ch_1*2**2), kernel_size=1, bias=bias)
        self.decoder_3 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[2])])
        self.up3_2 = Upsample(int(ch_1*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_2 = nn.Conv2d(int(ch_1*2**2), int(ch_1*2**1), kernel_size=1, bias=bias)
        self.decoder_2 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[1])])
        self.up2_1 = Upsample(int(ch_1*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_1 = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(transblocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(ch_1*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) \
            for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(ch_1*2**1), 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_4 = nn.Conv2d(int(ch_1*2**2), 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_2 = nn.Conv2d(int(ch_1*2**1), 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, rgb, nir): 
        B, C, H, W = rgb.shape
        rgbf = self.patch_embed(rgb)
        nirf = self.patch_embe_n(nir)

        inp_enc_level1 = rgbf
        out_enc_level1 = self.encoder_1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        #### for nir #####
        inp_enc_leveln1 = nirf
        out_enc_leveln1 = self.encoder_n1(inp_enc_leveln1)
        
        inp_enc_leveln2 = self.down1_n2(out_enc_leveln1)
        out_enc_leveln2 = self.encoder_n2(inp_enc_leveln2)

        inp_enc_leveln3 = self.down2_n3(out_enc_leveln2)
        out_enc_leveln3 = self.encoder_n3(inp_enc_leveln3) 

        inp_dec_level3 = self.up4_3(latent)
        enc_level3, enc_leveln3 = self.GMMs[2](out_enc_level3, out_enc_leveln3)
        enc_level3, enc_leveln3 = self.LMMs[2](enc_level3, enc_leveln3)
        inp_dec_level3 = torch.cat([inp_dec_level3, enc_level3 + enc_leveln3], 1)
        inp_dec_level3 = self.reduce_chan_3(inp_dec_level3)
        out_dec_level3 = self.decoder_3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        enc_level2, enc_leveln2 = self.GMMs[1](out_enc_level2, out_enc_leveln2)
        enc_level2, enc_leveln2 = self.LMMs[1](enc_level2, enc_leveln2)
        inp_dec_level2 = torch.cat([inp_dec_level2, enc_level2 + enc_leveln2], 1)
        inp_dec_level2 = self.reduce_chan_2(inp_dec_level2)
        out_dec_level2 = self.decoder_2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        enc_level1, enc_leveln1 = self.GMMs[0](out_enc_level1, out_enc_leveln1)
        enc_level1, enc_leveln1 = self.LMMs[0](enc_level1, enc_leveln1)
        inp_dec_level1 = torch.cat([inp_dec_level1, enc_level1 + enc_leveln1], 1)
        out_dec_level1 = self.decoder_1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out = rgb + self.output(out_dec_level1)
        rgb_input4 = F.interpolate(rgb, scale_factor=1/4)[:, :, :H//4, :W//4]
        out_4 = rgb_input4 + self.output_4(inp_dec_level3)[:, :, :H//4, :W//4]
        rgb_input2 = F.interpolate(rgb, scale_factor=1/2)[:, :, :H//2, :W//2]
        out_2 = rgb_input2 + self.output_2(inp_dec_level2)[:, :, :H//2, :W//2]
        return out, out_2, out_4




##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class TransGroup(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type, nb):
        super(TransGroup, self).__init__()

        TG = [TransformerBlock(dim, num_heads,ffn_expansion_factor, bias, LayerNorm_type) \
               for _ in range(nb)]

        TG.append(N.conv(dim, dim, mode='C'))

        self.rg = nn.Sequential(*TG)

    def forward(self, x):
        res = self.rg(x)
        return res + x

class LayerNormFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)