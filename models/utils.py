import torch
import torch.nn as nn


class MVN(nn.Module):

  def __init__(self, esp=1e-6):
    super(MVN, self).__init__()
    self.esp=esp

  def forward(self, x):
    mean = torch.mean(x, dim=(2,3), keepdim=True)
    std = torch.std(x, dim=(2,3), keepdim=True)
    x = (x-mean)/(std+self.esp)

    return x


class Conv_block(nn.Module):
    
  def __init__(self, in_c, out_c, filters=3, strides=1, padding=1, norm='mvn', reps=2):
    super(Conv_block, self).__init__()
    self.in_c = in_c
    self.out_c = out_c
    self.convs = nn.ModuleList()
    in_conv = self.in_c
    for i in range(reps):
      self.convs.append(nn.Conv2d(in_conv, self.out_c, filters, strides, padding=padding))
      if norm == 'mvn':
        self.convs.append(MVN())
      elif norm == 'bn':
        self.convs.append(nn.BatchNorm2d(self.out_c))
      elif norm == 'mvn+bn':
        self.convs.append(MVN())
        self.convs.append(nn.BatchNorm2d(self.out_c))
      self.convs.append(nn.ReLU(inplace=True))
      in_conv = self.out_c

  def forward(self, x):
    for layer in self.convs:
      x = layer(x)

    return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi