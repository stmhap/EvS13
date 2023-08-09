"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

import config as cfgfile

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    @staticmethod
    def get_act(atype, param=0.1):
        if atype == 'lrelu':
            return nn.LeakyReLU(param)
        elif atype == 'relu':
            return nn.ReLU()

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, act=cfgfile.ACTIVATION, bn_act=True,
                 dws=False):
        super(CNNBlock, self).__init__()

        bias = not bn_act
        layer_list = list()

        if dws and out_channels % in_channels == 0 and kernel_size == 3:
            layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, groups=in_channels,
                                        padding=padding, stride=stride, padding_mode='replicate'))
            if bn_act:
                layer_list.append(nn.BatchNorm2d(out_channels))
                layer_list.append(self.get_act(act))
            layer_list.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias, padding=0, stride=1))
        else:
            layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,
                                        stride=stride, padding_mode='replicate'))

        if bn_act:
            layer_list.append(nn.BatchNorm2d(out_channels))
            layer_list.append(self.get_act(act))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1, dws=False):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, 1),
                    CNNBlock(channels // 2, channels, 3, padding=1, dws=dws),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, rep):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, 3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, 1, bn_act=False),
            nn.AdaptiveAvgPool2d(cfgfile.S[rep])
        )
        self.num_classes = num_classes
        self.rep = rep

    def forward(self, x):
        x = self.pred(x)
        return x.reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            x_ = layer(x)
            if isinstance(layer, ScalePrediction):
                outputs.append(x_)
                continue

            x = x_
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        sp_rep = 0

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats, dws=True))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, 1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes, rep=sp_rep),
                    ]
                    sp_rep += 1
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers
