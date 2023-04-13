from collections import OrderedDict

import torch
import torch.nn as nn

from darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(filter_out)),
        ('relu', nn.LeakyReLU(0.1))
    ]))

def make_last_layers(filters_list, in_filter, out_filter):
    return nn.Sequential(
        conv2d(in_filter, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )


class YoloBody(nn.Module):
    def __init__(self, anchor_mask_list, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        # [64, 128, 256, 512, 1024]
        out_filters_list = self.backbone.layer_out_filters

        self.last_layer0 = make_last_layers([512, 1024], out_filters_list[-1], len(anchor_mask_list[0]) * (num_classes + 5))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters_list[-2] + 256, len(anchor_mask_list[1]) * (num_classes + 5))

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters_list[-3] + 128, len(anchor_mask_list[2]) * (num_classes + 5))

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        # 13,13,512 -> 13,13,1024->13,13,75
        out0 = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256 -> 26,26,768
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], dim=1)

        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        # 26,26,256 -> 26,26,512 -> 26,26,75
        out1 = self.last_layer1[5: ](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128 -> 52,52,374
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], dim=1)

        # 52,52,374 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        # 52,52,128 -> 52,52,256 -> 52,52,75
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2


