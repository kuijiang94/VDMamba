"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD
from mamba_ssm import Mamba
from monai.networks.layers.utils import get_act_layer, get_norm_layer
##########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# hidden_list = [256,256]
# hidden_list = [256,256,256,256]
hidden_list = [256,64,256]
L = 8
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class NRN(nn.Module):

    def __init__(self, in_dim, out_dim, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        imnet_in_dim = in_dim # 48#64
        imnet_out_dim = out_dim # 48#3
        
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2+4*L # attach coord
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim,imnet_out_dim,hidden_list)

    def query_rgb(self, inp, coord, cell=None):
        # inp torch.Size([8, 48, 96, 96])
        # coord torch.Size([1, 9216, 34])
        # cell torch.Size([1, 9216, 2])
        feat = inp
        # print('inp',inp.shape) torch.Size([1, 48, 128, 128])
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # print('feat_coord',feat_coord.shape) torch.Size([1, 2, 128, 128])
        # print('coord',coord.shape) torch.Size([1, 16384, 34])
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                '''
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                '''
                bs, q, h, w = feat.shape
                q_feat = feat.view(bs, q, -1).permute(0, 2, 1)

                bs, q, h, w = feat_coord.shape
                q_coord = feat_coord.view(bs, q, -1).permute(0, 2, 1)
                
                points_enc = self.positional_encoding(q_coord, L=L)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)  # [B,...,6L+3]
                '''
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                '''

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # print('inp1',inp.shape) # torch.Size([8, 9216, 466])

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    # print('rel_cell',rel_cell.shape) # torch.Size([1, 9216, 2])
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            
        bs, q, h, w = feat.shape
        ret = ret.view(bs, h, w,-1).permute(0, 3, 1, 2)
        return ret

    def forward(self, inp):
        bs, c, h, w = inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0)
        coord = coord.unsqueeze(0)
        
        points_enc = self.positional_encoding(coord,L=L)
        coord = torch.cat([coord, points_enc], dim=-1)  # [B,...,6L+3]
        # print('inp',inp.shape)
        # print('coord',coord.shape)
        # print('cell',cell.shape)
        NRN_seqs = []
        #print(inp.shape[0])
        for i in range(bs):
            fea = self.query_rgb(inp[i:i+1,:,:,:], coord, cell)
            NRN_seqs.append(fea)
            # print('query_rgb',fea.shape) torch.Size([1, 48, 128, 128])

        ret = torch.stack(NRN_seqs, dim=0).view(-1, c, h, w)
        # print('ret',ret.shape)
        return ret#self.query_rgb(inp, coord, cell)

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32).cuda()*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]

        return input_enc
##########################################################################
class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
    spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims==3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, norm: str, kernel_size: int, act: str= ("RELU", {"inplace": True})):
        # spatial_dims: int,
        # in_channels: int,
        # norm: tuple | str,
        # kernel_size: int,
        # act: tuple | str)
        # '''
        # Args:
            # spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            # in_channels: number of input channels.
            # norm: feature normalization type and arguments.
            # kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            # act: activation type and arguments. Defaults to ``RELU``.
        # '''
        super(ResMambaBlock, self).__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x
##########################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 浣跨敤鍝堝皵 haar 灏忔尝鍙樻崲鏉ュ疄鐜颁簩缁撮€嗗悜绂绘暎灏忔尝
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 浜岀淮绂绘暎灏忔尝
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 淇″彿澶勭悊锛岄潪鍗风Н杩愮畻锛屼笉闇€瑕佽繘琛屾搴︽眰

    def forward(self, x):
        return dwt_init(x)


# 閫嗗悜浜岀淮绂绘暎灏忔尝
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
##########################################################################
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)	
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

###########################################################################
### Gated-Dconv Feed-Forward Network (GDFN)
#class FeedForward(nn.Module):
#    def __init__(self, dim, ffn_expansion_factor, bias):
#        super(FeedForward, self).__init__()
#        self.act1 = nn.PReLU()
#        hidden_features = int(dim*ffn_expansion_factor)
#
#        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#    def forward(self, x):
#        x = self.project_in(x)
#        x1, x2 = self.dwconv(x).chunk(2, dim=1)
#        x = self.act1(x1) * x2
#        x = self.project_out(x)
#        return x
#
#
#
###########################################################################
### Multi-DConv Head Transposed Self-Attention (MDTA)
#class Attention(nn.Module):
#    def __init__(self, dim, num_heads, bias):
#        super(Attention, self).__init__()
#        self.num_heads = num_heads
#        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        
#        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        
#
#
#    def forward(self, k_fea, v_fea, q_fea):
#        b,c,h,w = q_fea.shape
#        q = self.q(q_fea)
#        k = self.k(k_fea)
#        v = self.v(v_fea)
#        #qkv = self.qkv_dwconv(self.qkv(x))
#        #q,k,v = qkv.chunk(3, dim=1)   
#        
#        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#        q = torch.nn.functional.normalize(q, dim=-1)
#        k = torch.nn.functional.normalize(k, dim=-1)
#
#        attn = (q @ k.transpose(-2, -1)) * self.temperature
#        attn = attn.softmax(dim=-1)
#
#        out = (attn @ v)
#        
#        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#        out = self.project_out(out)
#        return out
##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        #self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        #self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    # def forward(self, x):
        # b, c, h, w = x.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
    def forward(self, k_fea, v_fea, q_fea):
        b,c,h,w = q_fea.shape
        q = self.q(q_fea)
        k = self.k(k_fea)
        v = self.v(v_fea) 
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##  Sparse Transformer Block (STB) 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, in1, in2):
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))
        x = in2 + self.attn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))
        x = x + self.ffn(self.norm2(x))
        return x

###########################################################################
#class TransformerBlock(nn.Module):
#    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#        super(TransformerBlock, self).__init__()
#
#        self.norm_key = LayerNorm(dim, LayerNorm_type)
#        self.norm_query = LayerNorm(dim, LayerNorm_type)
#        self.norm_value = LayerNorm(dim, LayerNorm_type)
#        
#        self.attn = Attention(dim, num_heads, bias)
#        self.norm2 = LayerNorm(dim, LayerNorm_type)
#        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#
#    def forward(self, in1, in2):
#        # print('in1', in1.shape)
#        # print('in2', in2.shape)
#        # a = self.norm_key(in1)
#        # b = self.norm_query(in2)
#        # print('norm_key(in1)', a.shape)
#        # print('norm_query(in2)', b.shape)
#        x = in2 + self.attn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))
#        x = x + self.ffn(self.norm2(x))
#
#        return x
        
##########################################################################
class COBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(COBlock, self).__init__()

        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.COattn = Attention(dim, num_heads, bias)
        #self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, in1, in2):
        x = in2 + self.COattn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))

        return x
        
        
##########################################################################
class COBlock2(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(COBlock2, self).__init__()

        self.norm_key1 = LayerNorm(dim, LayerNorm_type)
        self.norm_query1 = LayerNorm(dim, LayerNorm_type)
        self.norm_value1 = LayerNorm(dim, LayerNorm_type)
        
        self.norm_key2 = LayerNorm(dim, LayerNorm_type)
        self.norm_query2 = LayerNorm(dim, LayerNorm_type)
        self.norm_value2 = LayerNorm(dim, LayerNorm_type)
        
        self.COattn1 = Attention(dim, num_heads, bias)
        self.COattn2 = Attention(dim, num_heads, bias)
        #self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, in1, in2, in3):
        x_12 = in2 + self.COattn1(self.norm_key1(in1),self.norm_value1(in1),self.norm_query1(in2))
        x_13 = in3 + self.COattn2(self.norm_key2(in1),self.norm_value2(in1),self.norm_query2(in3))
        return x_12, x_13
 ##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
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
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RestormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
        
##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
## Global context Layer
# class GCLayer(nn.Module):
    # def __init__(self, channel, reduction=16, bias=False):
        # super(GCLayer, self).__init__()
        # # global average pooling: feature --> point
        # #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # # feature channel downscale and upscale --> channel weight
        # self.conv_phi = nn.Conv2d(channel, 1, 1, stride=1,padding=0, bias=False)
        # self.softmax = nn.Softmax(dim=1)
		
        # self.conv_du = nn.Sequential(
                # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                # nn.Sigmoid()
        # )

    # def forward(self, x):
        # b, c, h, w = x.size()
        # #y = self.avg_pool(x)
        # y_1 = self.conv_phi(x).view(b, 1, -1).permute(0, 2, 1).contiguous()### b,hw,1
        # y_1_att = self.softmax(y_1)
        # print(y_1.size)
        # x_1 = x.view(b, c, -1)### b,c,hw
        # mul_context = torch.matmul(x_1, y_1_att)#### b,c,1
        # mul_context = mul_context.view(b, c, 1, -1)

        # y = self.conv_du(mul_context)
        # return x * y
		
##########################################################################
## Semantic-guidance Texture Enhancement Module
class STEM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=False):
        super(STEM, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 4
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = st_conv(3, n_feat, kernel_size, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(3, n_feat, kernel_size, bias=bias)
        self.former = TransformerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        self.conv_stem3 = conv(3, n_feat, kernel_size, bias=bias)
        self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)

    def forward(self, img_rain, res, img):
        #img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        #img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        #rain_mask = torch.sigmoid(res_fea)
        #rain_mask = self.CA_fea(res_fea)
        #att_fea = img_down * rain_mask + img_down
        att_fea = self.former(res_fea, img_down, img_down)
        img_fea = self.conv_stem3(img)
        S2FB2_FEA = self.S2FB(img_fea, att_fea)
        return S2FB2_FEA
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
		
class STEM_att(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(STEM_att, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 4
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        self.former = TransformerBlock(dim=n_feat//2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        # self.conv_stem3 = conv(n_feat, n_feat//2, kernel_size, bias=bias)
        # self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        self.conv_stem3 = conv(n_feat//2, n_feat, kernel_size=1, bias=bias)
        
    def forward(self, img_rain, res):
        #img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        #img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        #rain_mask = torch.sigmoid(res_fea)
        #rain_mask = self.CA_fea(res_fea)
        #att_fea = img_down * rain_mask + img_down
        att_fea = self.conv_stem3(self.former(res_fea, img_down))
        # img_fea = self.conv_stem3(img)
        # S2FB2_FEA = self.S2FB(img_fea, att_fea)
        return att_fea
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        
    
        # self.fuse_weight_ATOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # self.fuse_weight_RTOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # self.fuse_weight_ATOB.data.fill_(0.2)
        # # self.fuse_weight_RTOB.data.fill_(0.2)
        # self.conv_fuse_ATOB = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())
        
        # self.DSC = depthwise_separable_conv(n_feat, n_feat)
        # # self.DSC1 = depthwise_separable_conv(n_feat//2, n_feat)
        # #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        # self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        # #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    # def forward(self, x1, x2):
        # FEA_1to2 = self.DSC(x1*self.conv_fuse_ATOB(x2)*self.fuse_weight_ATOB)
        # #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        # #resin = FEA_1 + FEA_2
        # out= self.CA_fea(FEA_1to2) + x2
        # #res += resin
        # return out#x1 + resin
        
        
        self.DSC = depthwise_separable_conv(n_feat*2, n_feat)
        # self.DSC1 = depthwise_separable_conv(n_feat//2, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1,x2), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + x1
        #res += resin
        return res#x1 + resin

##########################################################################
## S2FB
class S2FB_4(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_4, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC = depthwise_separable_conv(n_feat*4, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*3, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_1 = self.DSC(torch.cat((x1, x2,x3,x4), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x2,x3,x4), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + FEA_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res

class S2FB_p(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_p, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC1 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC2 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC3 = depthwise_separable_conv(n_feat*2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_34 = self.DSC1(torch.cat((x3, x4), 1))
        FEA_34_2 = self.DSC2(torch.cat((x2, FEA_34), 1))
        FEA_34_2_1 = self.DSC3(torch.cat((x1, FEA_34_2), 1))
        res= self.CA_fea(FEA_34_2_1) + FEA_34_2_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res


class DownSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
        
class UpSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
##########################################################################
class h_sigmoid(nn.Module):
    #def __init__(self, inplace=True):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
############  V (H) attention Layer
class CoordAtt_V(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_V, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))#nn.AdaptiveAvgPool2d((None, 1)),for training

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h_pool = self.pool_h(x)##n,c,h,1
        y = self.conv1(x_h_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
		
        a_h_att = self.conv_h(y_act).sigmoid()


        out = identity * a_h_att

        return out
############  H (W) attention Layer
class CoordAtt_H(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_H, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        #x_h = self.pool_h(x)
        x_w_pool = self.pool_w(x).permute(0, 1, 3, 2)##n,c,W,1
        y = self.conv1(x_w_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
        
        y_act_per = y_act.permute(0, 1, 3, 2)
        a_w_att = self.conv_w(y_act_per).sigmoid()

        out = identity * a_w_att

        return out
        
class Composition(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(Composition, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # mip = max(8, inp // 4)

        self.conv1 = nn.Sequential(nn.Conv2d(inp*2, oup, kernel_size=1, stride=1, padding=0), CAB_dsc(oup, kernel_size=3, reduction=4, bias=False, act=nn.GELU()))
        self.bn1 = nn.BatchNorm2d(oup)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, v, h):
        # identity = x
        
        # n,c,h,w = x.size()
        # x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([v, h], dim=1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 

        a_v = self.conv_h(y).sigmoid()
        a_h = self.conv_w(y).sigmoid()
        # print("a_v",a_v.shape)
        # print("a_h",a_h.shape)
        # print("y",y.shape)
        
        out = a_h*h  + a_v*v

        return out
##########################################################################
## Coupled Representaion Module (CRM)
class CRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(CRM, self).__init__()
        #num_blocks = num_cab
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        #modules_body = []
        #self.CAB = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.down1 = DownSample(n_feat)
        #self.down2 = DownSample(n_feat)
        spatial_dims = 3
        norm: tuple | str = ("GROUP", {"num_groups": 8})
        self.V_Mamba_fusion = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))#NRN(n_feat, n_feat)
        self.H_Mamba_fusion = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))#NRN(n_feat, n_feat)
        
        self.V_ATT_rain = CoordAtt_V(n_feat, n_feat)
        self.H_ATT_rain = CoordAtt_H(n_feat, n_feat)
        
        self.V_fea_rain = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.H_fea_rain = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        
        self.V_ATT = CoordAtt_V(n_feat, n_feat)
        self.H_ATT = CoordAtt_H(n_feat, n_feat)
        self.V_fea = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))
        self.H_fea = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))
        
        # self.CAB1 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        # self.CAB2 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB3 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB4 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        # self.CAB5 = nn.Sequential(conv(n_feat*2, n_feat, kernel_size, bias=bias), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))#nn.Sequential(conv(n_feat*2, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2, kernel_size, reduction, bias=bias, act=act), conv(n_feat//2, n_feat, kernel_size, bias=bias))#, CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act))
        #self.former = TransformerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock_1 = COBlock(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock_2 = COBlock(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock2 = COBlock2(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        # self.up1 = UpSample(n_feat)
        # self.up2 = UpSample(n_feat)
        # self.STEM_att12 = STEM_att(n_feat, kernel_size, bias=bias)
        # self.STEM_att21 = STEM_att(n_feat, kernel_size, bias=bias)
        # self.S2FB2_1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        # self.S2FB2_2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        # self.S2FB2_3 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        # self.S2FB2_4 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.S2FB2_3 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.conv = conv(n_feat, n_feat, kernel_size)
        #self.body = nn.Sequential(*modules_body) S2FB_2
        self.Composition     = Composition(n_feat, n_feat)
    def forward(self, x):
        # x0 = x[0]  ### V
        # x1 = x[1]  ### H

        xV_r_fea  = self.V_Mamba_fusion(self.V_ATT_rain(x))
        xH_r_fea  = self.H_Mamba_fusion(self.H_ATT_rain(x))
        
        # xV_b_fea  = x0 - xV_r_fea
        # xH_b_fea  = x1 - xH_r_fea
        V2H_ATT_fea = self.H_fea(self.H_ATT(xV_r_fea))
        H2V_ATT_fea = self.V_fea(self.V_ATT(xH_r_fea))
        
        # print('res12', res12.shape)
        H_fea = self.CAB3(V2H_ATT_fea) + xH_r_fea ### back
        V_fea = self.CAB4(H2V_ATT_fea) + xV_r_fea ### rain
        
        #x1v2 = res + x1v1
        # x[0] = V_fea #+ x1 #self.up1(x1v2) + x1
        # x[1] = H_fea #+ x2 #self.up2(x2v1) + x2
        # x[2] = x2v2 #+ x2 #self.up2(x2v1) + x2
        rain_res = self.Composition(V_fea, H_fea)
        return x - rain_res

##########################################################################
class COmodule(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(COmodule, self).__init__()
        modules_body = []
        modules_body = [CRM(n_feat, kernel_size, reduction, act, bias) for i in range(num_cab)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x1):
        res = self.body(x1)
        return res
        
##########################################################################
## Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.iwt_rain = IWT()
        self.iwt_back = IWT()
        self.iwt_res = IWT()
        self.recon_B =  conv(12, 12, kernel_size, bias=bias)
        self.recon_R = conv(12, 12, kernel_size, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(16, 32, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 12, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        xB = x[0]
        xR = x[1]

        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        # res = self.avg_pool(recon_B + recon_R)
        # res_att = self.conv_du(res)
        # re_rain = xB*res_att + xR*(1-res_att)
        #rain_img = self.iwt_rain(re_rain)
        #back_img = self.iwt_back(xB)
        # rain_img = self.iwt_rain(re_rain)
        rain_res = self.iwt_res(recon_R)
        back_img = self.iwt_back(xB)
        return [back_img, rain_res]
        
        
##########################################################################
class MAMNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_orb, num_cab):
        super(MAMNet, self).__init__()
        
        self.dwt = DWT()
        self.iwt_back = IWT()
        #self.iwt_rain = IWT()
        #self.iwt_back = IWT()
        act=nn.GELU()#nn.ReLU()
        #num_blocks = 1
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        norm: tuple | str = ("GROUP", {"num_groups": 8})
        spatial_dims = 3
        # self.down_1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        # self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2 = nn.Sequential(conv(3*4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        # self.fuse_conv  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        #self.former = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act))
        #self.down_R = st_conv(n_feat, n_feat*2, kernel_size, bias=bias)
        #self.down_B = st_conv(n_feat, n_feat*2, kernel_size, bias=bias)
        #self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_orb)
        # self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))#, RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        # self.shallow_feat1_1 = conv(n_feat//2, n_feat, kernel_size, bias=bias)#, RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        # self.shallow_feat2_1 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2_2 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2_3 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(4*3, n_feat, kernel_size, bias=bias), CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), ResMambaBlock(spatial_dims, n_feat, norm, kernel_size))
        
        self.V_ATT = CoordAtt_V(n_feat, n_feat)
        self.H_ATT = CoordAtt_H(n_feat, n_feat)
        
        self.V_fea = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.H_fea = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        
        # self.shallow_feat_R = conv(n_feat, n_feat, kernel_size, bias=bias)
        # self.shallow_feat_B = conv(n_feat, n_feat, kernel_size, bias=bias)
        #self.fuse_conv  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        #self.former = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        
        self.comodule = COmodule(n_feat, kernel_size, reduction, act, bias, num_cab)
        #self.UP_B = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))
        #self.UP_R = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))

        #self.tail_LL_r = conv(n_feat, 3, kernel_size, bias=bias)
        
        # self.cat_rain = nn.Sequential(CAB_dsc(n_feat*2, kernel_size, reduction, bias=bias, act=act), conv(n_feat*2, 3*4, kernel_size, bias=bias)) 
        # self.cat_back = nn.Sequential(CAB_dsc(n_feat*2, kernel_size, reduction, bias=bias, act=act), conv(n_feat*2, 3*4, kernel_size, bias=bias)) 
        
        #conv(n_feat*2, 3*4, kernel_size, bias=bias)
        #self.tail_LL_r = conv(n_feat*2, 3*4, kernel_size, bias=bias)
        #self.tail_HH_b = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_V_b = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_H_b = conv(n_feat, 3, kernel_size, bias=bias)
        
        #self.tail_HH_r = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_V_r = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_H_r = conv(n_feat, 3, kernel_size, bias=bias)
        
        #self.S2FB_fuse = S2FB_2(n_feat, reduction, bias=bias, act=act)
        # self.recon = RRB(n_feat, kernel_size, act, bias=bias)
        #self.tail     = conv(n_feat, 3, kernel_size, bias=bias)
        self.Composition     = Composition(n_feat, n_feat)
        self.tail     = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), conv(n_feat, 4*3, kernel_size, bias=bias))
        
    def forward(self, x):

        #H = x.size(2)
        #W = x.size(3)
        dwt_fea = self.dwt(x)
        
        x_LL = dwt_fea[0]
        x_V = dwt_fea[1]
        x_H = dwt_fea[2]
        x_HH = dwt_fea[3]
        # Two Patches for Stage 2
        #xtop_img  = x[:,:,0:int(H/2),:]
        #xbot_img  = x[:,:,int(H/2):H,:]
		
        #x2_img_down = self.down_1(x)
        #x2_img_down_fea = self.shallow_feat1(x2_img_down)
		
        # Four Patches for Stage 1
        # x1ltop_img = xtop_img[:,:,:,0:int(W/2)]
        # x1rtop_img = xtop_img[:,:,:,int(W/2):W]
        # x1lbot_img = xbot_img[:,:,:,0:int(W/2)]
        # x1rbot_img = xbot_img[:,:,:,int(W/2):W]
        
        # stage1_input = torch.cat([x1ltop_img, x1rtop_img, x1lbot_img, x1rbot_img],1) 
        # x1fea = self.shallow_feat2(stage1_input)
        # former_fea = self.former(x1fea)
        
        
        # stage1_fuse = torch.cat([x2_img_down_fea, former_fea],1) 
        # fuse_fea = self.fuse_conv(stage1_fuse)
        
        # x_LL_fea = self.shallow_feat1(x_LL)
        # x_LL_fea = self.shallow_feat1_1(x_LL_fea)
        #LL_fea = self.orsnet(x_LL_fea)

        # x_V_fea = self.shallow_feat2_1(x_V) 
        # x_H_fea = self.shallow_feat2_2(x_H) 
        # x_HH_fea = self.shallow_feat2_3(x_HH) 
        stage_fuse = torch.cat([x_LL, x_V, x_H, x_HH],1)

        x_fuse_fea = self.shallow_feat2(stage_fuse)
        
        # former_fea = self.former(x_fuse_fea)
        #R_down = self.down_R(former_fea)
        #B_down = self.down_B(former_fea)
        
        # v_ATT_fea = self.V_fea(self.V_ATT(x_fuse_fea))
        # h_ATT_fea = self.H_fea(self.H_ATT(x_fuse_fea))
        
        # xB_fea = self.shallow_feat_B(former_fea)
        # xR_fea = self.shallow_feat_R(former_fea)

        x_fuse_fea_out = self.comodule(x_fuse_fea)
        # Composition_fea = self.Composition(v_ATT_fea_out, h_ATT_fea_out)
        #x_V_rain = self.tail_V_r(rain_fea)
        #x_HH_rain = self.tail_HH_r(rain_fea)
        #x_H_rain = self.tail_H_r(rain_fea)
        #x_LL_rain = self.tail_LL_r(LL_fea)
        
        # b_cat = self.cat_back(torch.cat((x_LL_b, back_fea), 1))
        # r_cat = self.cat_rain(torch.cat((rain_fea,x_LL_fea-x_LL_b), 1))
        
        #x_V_b = self.tail_V_b(back_fea)
        #x_HH_b = self.tail_HH_b(back_fea)
        #x_H_b = self.tail_H_b(back_fea)
        #fea_B = self.UP_B(back_fea)
        #fea_R = self.UP_R(rain_fea)
        #fused_fea = self.S2FB_fuse(xGE_fea, xLE_fea)
        #fused_fea = self.S2FB_fuse(or_fea, fused_fea)
        #b_cat = torch.cat((x_LL-x_LL_rain, x_V_b, x_H_b, x_HH_b), 1)
        #r_cat = torch.cat((x_LL_rain, x_V_rain, x_H_rain, x_HH_rain), 1)
        
        #rain_img = self.iwt_rain(r_cat)
        #back_img = self.iwt_back(b_cat)
        
        # [img_B, img_R] = self.recon([b_cat, r_cat])
        #recon_up  = shuffle_up(fused_fea, 2)

        out = self.tail(x_fuse_fea_out)
        out_img = self.iwt_back(out)
        return out_img
		
##########################################################################
class WaveMAMBA(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_orb=3, num_cab=10, bias=False):
        super(WaveMAMBA, self).__init__()

        act=nn.PReLU()
        self.MAMNet = MAMNet(n_feat, kernel_size, reduction, act, bias, num_orb, num_cab)

        
    def forward(self, x_img): #####b,c,h,w
        #print(x_img.shape)

        imitation = self.MAMNet(x_img)
        # print("x_img",x_img.device)
        # print("rain_res",rain_res.device)
        # print("imitation",imitation.device)
        #imitation = x_img - res
        return imitation#[imitation, x_img - imitation]