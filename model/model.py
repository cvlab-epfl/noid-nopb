import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

from misc.utils import pin_memory, to_device


class PinnableDict(): #collections.UserDict
    def __init__(self, org_data):
        super().__init__()
        self.data = org_data
    
    def __getitem__(self, _idx):
        return self.data[_idx]

    def __setitem__(self, _idx, val):
        self.data[_idx] = val

    def pin_memory(self):
        return PinnableDict(pin_memory(self.data))

    def to(self, *args, **kwargs):
        return PinnableDict(to_device(self.data, *args, **kwargs))

    def keys(self):
        return self.data.keys()

    # #Pass undefined attribute/method to data object
    # def __getattr__(self, name):
    #     return getattr(self.data, name)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c)
        
    def forward(self, inputs):
        x = self.up(inputs)
        x = torch.cat([x], axis=1)
        x = self.conv(x)
        return x
    
class MotionModel(torch.nn.Module):
    def __init__(self, model_conf, data_conf):
        super(MotionModel, self).__init__()


        backbone_out_channels = 512

        if model_conf["backbone"] == "r50":
            self.backbone = models.resnet50(weights='DEFAULT')
        elif model_conf["backbone"] == "r18":
            self.backbone = models.resnet18(weights='DEFAULT')
            backbone_out_channels = 128
        elif model_conf["backbone"] == "r34":
            self.backbone = models.resnet34(weights='DEFAULT')
            backbone_out_channels = 128
        elif model_conf["backbone"] == "r101":
            self.backbone = models.resnet101(weights='DEFAULT')
        elif model_conf["backbone"] == "r152":
            self.backbone = models.resnet152(weights='DEFAULT')
        else:
            raise ValueError("Backbone not supported")
        
        if data_conf["reid_feature"]:
            input_channels = 512 + 1
        else:
            input_channels = 1

        new_fl = torch.nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        new_fl.weight.tensor = self.backbone._modules["conv1"].weight.mean(dim=1).unsqueeze(1)
        self.backbone._modules["conv1"] = new_fl

        #remove last layer of resnet
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-4])
        
        self.fusion_layers = torch.nn.Sequential(
            decoder_block(backbone_out_channels*2, backbone_out_channels),
            decoder_block(backbone_out_channels, backbone_out_channels),
            decoder_block(backbone_out_channels, backbone_out_channels)
        )


        self.motion_reg = torch.nn.Sequential(
                self.fusion_layers,
                torch.nn.ReLU(), #since the last layer of pretrained model r34 is convolution
                torch.nn.Conv2d(backbone_out_channels, backbone_out_channels//2, 1), 
                torch.nn.ReLU(),                                       
                torch.nn.Conv2d(backbone_out_channels//2, 2, 1, bias=False),
              )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_data):
        
        x0 = input_data["hm_0"]
        x1 = input_data["hm_1"]
       
        x0 = self.backbone(x0)
        x1 = self.backbone(x1)
        
        x_forw = torch.cat((x0, x1), dim=1)
        x_forw = self.motion_reg(x_forw)
        
        x_back = torch.cat((x1, x0), dim=1)
        x_back = self.motion_reg(x_back)

        return PinnableDict({"offset_0_1f": x_forw, "offset_1_0b": x_back, "time_stats":{}})