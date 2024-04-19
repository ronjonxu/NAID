import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
import math
# using NAFNet structure
class NIRNAFModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(NIRNAFModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['NAF_MSE', 'NAF_MSE2', 'NAF_MSE4', 'NAF_MSE8', 'Total']
        self.visual_names = ['data_ir', 'data_gt', 'data_gt_noise', 'data_out'] 
        self.model_names = ['NAF'] 
        self.optimizer_names = ['NAF_optimizer_%s' % opt.optimizer]

        naf = nirnaf(opt, width=32, enc_blk_nums=[2, 2, 4, 8],middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
        self.netNAF= N.init_net(naf, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:		
            self.optimizer_NAF = optim.Adam(self.netNAF.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)
        
            self.optimizers = [self.optimizer_NAF]

            self.criterionMSE = N.init_net(MSELoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input, epoch):
        self.data_ir = input['ir_img'].to(self.device)
        self.data_gt = input['gt_img'].to(self.device)
        self.data_gt_noise = input['noise_img'].to(self.device)
        self.epoch = epoch
        self.image_paths = input['fname']

    def forward(self):
        if self.isTrain:
            if self.epoch != -1: 
                self.data_out, self.out = self.netNAF(self.data_gt_noise, self.data_ir)
            else:
                self.data_out, self.out = self.netNAF(self.data_gt_noise, self.data_ir)
        else:
            self.data_out, self.out = self.netNAF(self.data_gt_noise,  self.data_ir)

    def backward(self):
        _, _, H, W = self.data_gt.shape
        data_gt2 = F.interpolate(self.data_gt, scale_factor=0.5, mode='bilinear')[:, :, :H//2, :W//2]
        data_gt4 = F.interpolate(self.data_gt, scale_factor=0.25, mode='bilinear')[:, :, :H//4, :W//4]
        data_gt8 = F.interpolate(self.data_gt, scale_factor=0.125, mode='bilinear')[:, :, :H//8, :W//8]
        self.loss_NAF_MSE = self.criterionMSE(self.data_out, self.data_gt).mean()
        self.loss_NAF_MSE2 = self.criterionMSE(self.out[2], data_gt2).mean()
        self.loss_NAF_MSE4 = self.criterionMSE(self.out[1], data_gt4).mean()
        self.loss_NAF_MSE8 = self.criterionMSE(self.out[0], data_gt8).mean()
        self.loss_Total = self.loss_NAF_MSE + self.loss_NAF_MSE2 + self.loss_NAF_MSE4 + self.loss_NAF_MSE8
        self.loss_Total.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_NAF.zero_grad()
        self.backward()
        self.optimizer_NAF.step()
    


class GMM(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv3_1 = N.conv(c, c, kernel_size=1, padding=0, mode='C')
        self.conv5_1 = N.conv(c, c, kernel_size=1, padding=0, mode='C')
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
        feat_1 = self.conv3_1(rgb)
        feat_2 = self.conv5_1(nir)
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


class nirnaf(nn.Module):

    def __init__(self, opt, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.opt = opt
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro_nir = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.encoders_nir = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs_nir = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.GMMs = nn.ModuleList()
        self.LMMs = nn.ModuleList()
        self.multiout_convs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )

            self.encoders_nir.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs_nir.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            self.GMMs.append(GMM(chan))
            self.LMMs.append(LMM(chan))
            self.multiout_convs.append( N.conv(chan, 3, 3,1,1, bias=True, mode="C"))

            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, noise, nir):
        B, C, H, W = noise.shape
        noise = self.check_image_size(noise)
        nir = self.check_image_size(nir)

        x = self.intro(noise)
        y = self.intro_nir(nir)

        encs = []
        encs_nir = []
        out = []

        for encoder, encoder_nir, down, down_nir in zip(self.encoders, self.encoders_nir, self.downs, self.downs_nir):
            x = encoder(x)
            y = encoder_nir(y)
            encs.append(x)
            encs_nir.append(y)
            x = down(x)
            y = down_nir(y)

        x = self.middle_blks(x)

        i = 0
        for decoder, up, enc_skip, enc_nir_skip in zip(self.decoders, self.ups, encs[::-1], encs_nir[::-1]):
            x = up(x)
            feat_rgb, feat_nir = self.GMMs[i](enc_skip, enc_nir_skip)
            feat_rgb, feat_nir = self.LMMs[i](feat_rgb, feat_nir)
            x = x + feat_rgb + feat_nir
            x = decoder(x)
            if i < 3:
                temp_input = F.interpolate(noise, scale_factor=1/2**(3-i))
                temp_out = self.multiout_convs[i](x) + temp_input
                out.append(temp_out[:, :, :H//2**(3-i), :W//2**(3-i)])
            i += 1

        x = self.ending(x)
        x = x + noise
        return x[:, :, :H, :W], out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# -------------------------------------------------------
# NAFBlcok
# -------------------------------------------------------

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
    
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

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