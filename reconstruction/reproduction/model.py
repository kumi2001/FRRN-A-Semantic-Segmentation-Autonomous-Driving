"""Compact FRRN-style two-stream segmenter: a NEW reimplementation, not recovered May 2026 source."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, channels_in:int, channels_out:int, kernel_size:int=3):
        super().__init__(nn.Conv2d(channels_in,channels_out,kernel_size,padding=kernel_size//2,bias=False),
                         nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True))

class ResidualUnit(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.conv1=ConvBNReLU(channels,channels)
        self.conv2=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return F.relu(x+self.conv2(self.conv1(x)))

class FullResolutionResidualUnit(nn.Module):
    """Fuse full-resolution z and pooled-resolution y streams."""
    def __init__(self,pool_ch:int,full_ch:int,inner_ch:int):
        super().__init__()
        self.fuse=ConvBNReLU(pool_ch+full_ch,inner_ch)
        self.refine=ConvBNReLU(inner_ch,pool_ch)
        self.residual=nn.Conv2d(pool_ch,full_ch,1)
    def forward(self,y:torch.Tensor,z:torch.Tensor):
        hz,wz=z.shape[-2:]; yy,wy=y.shape[-2:]
        down_z=F.interpolate(z,size=(yy,wy),mode="bilinear",align_corners=False)
        h=self.refine(self.fuse(torch.cat((y,down_z),dim=1)))
        y=y+h
        z=z+F.interpolate(self.residual(h),size=(hz,wz),mode="bilinear",align_corners=False)
        return y,z

class FRRNStyle(nn.Module):
    """A small FRRN-inspired two-stream model, NOT exact FRRN-A architecture or weights."""
    def __init__(self,num_classes:int=4,width:int=16):
        super().__init__()
        if num_classes<2 or width<4:
            raise ValueError("num_classes must be >=2 and width >=4")
        self.num_classes,self.width=num_classes,width
        self.stem=ConvBNReLU(3,width,5)
        self.initial=ResidualUnit(width)
        self.pooled=ConvBNReLU(width,width*2)
        self.blocks=nn.ModuleList([FullResolutionResidualUnit(width*2,width,width*2) for _ in range(3)])
        self.head=nn.Sequential(ConvBNReLU(width*3,width),nn.Conv2d(width,num_classes,1))
    def forward(self,images:torch.Tensor)->torch.Tensor:
        if images.ndim!=4 or images.shape[1]!=3:
            raise ValueError("expected [batch,3,height,width]")
        if min(images.shape[-2:])<16:
            raise ValueError("minimum spatial dimensions 16x16")
        z=self.initial(self.stem(images))
        y=self.pooled(F.avg_pool2d(z,4))
        for unit in self.blocks:y,z=unit(y,z)
        y=F.interpolate(y,size=z.shape[-2:],mode="bilinear",align_corners=False)
        return self.head(torch.cat((y,z),dim=1))
