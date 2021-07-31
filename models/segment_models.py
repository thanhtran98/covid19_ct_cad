import torch
import torch.nn as nn
from .utils import Conv_block, Attention_block

class Unet(nn.Module):

  def __init__(self, n_class, n_channel, norm='mvn', up_op='bilinear'):
    super(Unet, self).__init__()
    self.n_class = n_class
    self.norm = norm
    self.maxpool = nn.MaxPool2d(2,2)
    if up_op == 'bilinear':
      self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    # elif up_op == 'transpose':
    #   self.upsample = nn.ConvTranspose2d
    self.encoder = nn.ModuleList([
                                    Conv_block(n_channel, 64, norm = self.norm), self.maxpool,
                                    Conv_block(64, 128, norm= self.norm), self.maxpool,
                                    Conv_block(128, 256, norm= self.norm), self.maxpool,
                                    Conv_block(256, 512, norm= self.norm), self.maxpool
    ])
    self.bridge = Conv_block(512, 1024, norm= self.norm)
    self.decoder = nn.ModuleList([
                                    self.upsample, Conv_block(512+1024, 512, norm=self.norm),
                                    self.upsample, Conv_block(256+512, 256, norm=self.norm),
                                    self.upsample, Conv_block(128+256, 128, norm=self.norm),
                                    self.upsample, Conv_block(64+128, 64, norm=self.norm)
    ])
    self.output = nn.Sequential(nn.Conv2d(64, self.n_class, 1), nn.Sigmoid())

  def forward(self, x):
    fms = []
    for i, block in enumerate(self.encoder):
      if i%2 == 0:
        fms.append(block(x))
      else:
        x = block(fms[-1])
    x = self.bridge(x)
    for i, block in enumerate(self.decoder):
      x = block(x)
      if i%2 == 0:
        x = torch.cat([fms[-1-(i//2)], x], dim=1)
    x = self.output(x)

    return x