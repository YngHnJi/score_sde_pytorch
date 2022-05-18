import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import utils, layers, layerspp, normalization
#import layers # debug purpose

NIN = layers.NIN

default_init = layers.default_init
get_act = layers.get_act

#modified from ResnetBlockBigGANpp
class ResMLPBlock(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    #self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 256), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    #self.Conv_0 = conv3x3(in_ch, out_ch)
    self.Linear_0 = nn.Linear(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    #self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 256), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Linear_1 = nn.Linear(out_ch, out_ch)
    if in_ch != out_ch or up or down:
      #self.Conv_2 = conv1x1(in_ch, out_ch)
      self.Linear_2 = nn.Linear(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    #h = self.Conv_0(h)
    h = self.Linear_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      #h += self.Dense_0(self.act(temb))[:, :, None, None]
      h += self.Dense_0(self.act(temb))
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    #h = self.Conv_1(h)
    h = self.Linear_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      #x = self.Conv_2(x)
      x = self.Linear_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class AttnBlock_NLP(nn.Module):
  def __init__(self, channels, skip_rescale=False, reduction_ratio=4):
    super().__init__()
    self.channels = channels
    self.skip_rescale = skip_rescale
    self.reduction_channel = channels // reduction_ratio

    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=self.channels, eps=1e-6)
    self.k_linear = nn.Linear(self.channels, self.reduction_channel)
    self.q_linear = nn.Linear(self.channels, self.reduction_channel)
    self.v_linear = nn.Linear(self.channels, self.reduction_channel)
    self.restore_linear = nn.Linear(self.reduction_channel, self.channels)
    
    self.softmax = nn.Softmax(dim = 2)

  def forward(self, x):
    #batch_size = x.shape[0]
    x = self.GroupNorm_0(x)
    k = self.k_linear(x)
    q = self.q_linear(x)
    w = torch.einsum('b i d, b d j -> b i j', k.unsqueeze(2), q.unsqueeze(1))
    w = self.softmax(w)

    v = self.v_linear(x)

    out = torch.bmm(w, v.unsqueeze(2)).squeeze()
    out = out / (self.reduction_channel ** 0.5)
    out = self.restore_linear(out)

    if not self.skip_rescale:
      return x + out
    else:
      return (x + out) / np.sqrt(2.)

if __name__=="__main__":
  # Block = ResMLPBlock(act=nn.SiLU(), in_ch=1024, out_ch=1024)
  # temp = torch.randn(128, 1024)

  # result = Block(temp)
  # print("debug")
  temp_attn = AttnBlock_NLP(1024, reduction_ratio=4)
  dummy = torch.randn(128, 1024)
  out = temp_attn(dummy)
