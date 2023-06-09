import math
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        # input_size = (600, 600, 3)
        self.inplanes = 64
        super(ResNet, self).__init__()

        # (600, 600, 3) -> (300, 300, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # (300, 300, 64) -> (150, 150, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # (150, 150, 64) -> (150, 150, 256)
        self.layer1 = self._make_layers(block, 64, layers[0])

        # (150, 150, 256) -> (75, 75, 512)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)

        # (75, 75, 512) -> (38, 38, 1024)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        # self.layer4 for classifier
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 4])

    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

if __name__ == "__main__":
    features, classifier = resnet50()
    if os.path.exists('runs/test_ResNet50'):
        shutil.rmtree('runs/test_ResNet50')
    writer = SummaryWriter('runs/test_ResNet50')
    x = torch.randn(1, 3, 600, 600)
    writer.add_graph(features, x)
    x = features(x)
    print(x.shape)
    writer.add_graph(classifier, x)