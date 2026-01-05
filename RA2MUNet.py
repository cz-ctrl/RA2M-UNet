import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import A2M,GA2M
import eca_layer

class RepModuleWithCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepModuleWithCA, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.ca = eca_layer()
        self.bn=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)

        self.fused_conv = None
    def forward(self, x):
        if self.fused_conv is None:
            out = self.conv1x1(x) + self.conv3x3(x)
            out=self.bn(out)
            out=self.relu(out)
            out= out *self.ca(out)
            return out 
        else:
            out = self.fused_conv(x)
            out= out *self.ca(out)
            return out 

class RepModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.fused_conv = None

    def forward(self, x):
        if self.fused_conv is None:
            out = self.conv1x1(x) + self.conv3x3(x)
            out=self.bn(out)
            out=self.relu(out)
            return out
        else:
            out = self.fused_conv(x)
            out=self.bn(out)
            out=self.relu(out)
            return out
class Base_C(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Base_C, self).__init__()
        self.down1=RepModule(in_channels,out_channels)       
    def forward(self,x):
        x=self.down1(x)
        return x   
  
class Double_CA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_CA, self).__init__()
        
        self.down1=RepModule(in_channels,out_channels)
        self.down2=RepModuleWithCA(out_channels,out_channels)
        
    def forward(self,x):
        x=self.down1(x)
        x=self.down2(x)
        return x


class RA2MUNet(nn.Module):
    def __init__(self, in_ch, out_ch, gt_ds=True):
        super(RA2MUNet, self).__init__()
        
        f_list = [8, 16, 32, 64, 96, 128]
        
        # Encoder
        self.encoder1 = Base_C(in_ch, f_list[0])
        self.encoder2 = Base_C(f_list[0], f_list[1])
        self.encoder3 = Base_C(f_list[1], f_list[2])
        self.encoder4 = Double_CA(f_list[2], f_list[3])
        self.encoder5 = Double_CA(f_list[3], f_list[4])
        self.encoder6 = Double_CA(f_list[4], f_list[5])
        self.bottleneck = A2M(f_list[5])
        
        # Decoder
        self.decoder6 = Double_CA(f_list[5], f_list[4])
        self.decoder5 = Double_CA(f_list[4] * 2, f_list[3])
        self.decoder4 = Double_CA(f_list[3] * 2, f_list[2])
        self.decoder3 = Base_C(f_list[2] * 2, f_list[1])
        self.decoder2 = Base_C(f_list[1] * 2, f_list[0])
        self.decoder1 = Base_C(f_list[0] * 2, f_list[0])
        
        # Output layer
        self.output = nn.Conv2d(f_list[0], out_ch, kernel_size=1)
        
        self.att_down_1 = A2M(f_list[1])
        self.att_down_2 = A2M(f_list[2])
        self.att_down_3 = A2M(f_list[3])
        self.att_down_4 = A2M(f_list[4])
        self.att_down_5 = A2M(f_list[5])

        self.att_up_1 = A2M(f_list[1])
        self.att_up_2 = A2M(f_list[2])
        self.att_up_3 = A2M(f_list[3])
        self.att_up_4 = A2M(f_list[4])
        
        self.me_mb5 = GA2M(f_list[5], f_list[4])
        self.me_mb4 = GA2M(f_list[4], f_list[3])
        self.me_mb3 = GA2M(f_list[3], f_list[2])
        self.me_mb2 = GA2M(f_list[2], f_list[1])

        self.gt_ds = gt_ds
        
        if self.gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(f_list[0], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(f_list[1], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(f_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(f_list[3], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(f_list[4], 1, 1))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def _apply_attention(self, x, att_module, downsample=False):
        if downsample:
            x = F.max_pool2d(x, 2)
        x = torch.permute(x, (0, 2, 3, 1))
        x = att_module(x)
        x = torch.permute(x, (0, 3, 1, 2))
        return x
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc2 = self._apply_attention(enc2, self.att_down_1, downsample=True)
        
        enc3 = self.encoder3(enc2)
        enc3 = self._apply_attention(enc3, self.att_down_2, downsample=True)
        
        enc4 = self.encoder4(enc3)
        enc4 = self._apply_attention(enc4, self.att_down_3, downsample=True)
        
        enc5 = self.encoder5(enc4)
        enc5 = self._apply_attention(enc5, self.att_down_4, downsample=True)
        
        enc6 = self.encoder6(enc5)
        enc6 = self._apply_attention(enc6, self.att_down_5, downsample=False)
        
        # Bottleneck
        bottleneck = self._apply_attention(enc6, self.bottleneck, downsample=False)
        
        dec6 = self.decoder6(bottleneck)
        dec6 = self._apply_attention(dec6, self.att_up_4, downsample=False)
        m5 = self.me_mb5(bottleneck, enc5)
        dec5 = torch.cat((dec6, m5), dim=1)
        dec5 = self.decoder5(dec5)
        dec5 = self._apply_attention(dec5, self.att_up_3, downsample=False)
        m4 = self.me_mb4(m5, enc4)
        dec4 = F.interpolate(dec5, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, m4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self._apply_attention(dec4, self.att_up_2, downsample=False)
        m3 = self.me_mb3(m4, enc3)
        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, m3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self._apply_attention(dec3, self.att_up_1, downsample=False)
        m2 = self.me_mb2(m3, enc2)
        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, m2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.output(dec1)
        if self.gt_ds:
            out1 = self.gt_conv1(dec2)
            out2 = self.gt_conv2(dec3)
            out3 = self.gt_conv3(dec4)
            out4 = self.gt_conv4(dec5)
            out5 = self.gt_conv5(dec6)
            out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, scale_factor=16, mode='bilinear', align_corners=True)
            out5 = F.interpolate(out5, scale_factor=16, mode='bilinear', align_corners=True)
            
            return output, out1, out2, out3, out4, out5
        
        return output


