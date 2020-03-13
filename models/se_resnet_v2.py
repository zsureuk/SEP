import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch

__all__ = ['SEP', 'se_resnet20', 'se_resnet32', 'se_resnet44', 'se_resnet56', 'se_resnet110']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # for i in range(3):
        #     if i == 3:     
        if planes == 16:
            self.globalAvgPool = nn.AvgPool2d(32, stride=1)
            # self.globalAvgPool1 = nn.AvgPool2d(64, stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d(16, stride=1)
            self.globalAvgPool1 = nn.AvgPool2d(32, stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d(8, stride=1)
            self.globalAvgPool1 = nn.AvgPool2d(16, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes)
        self.fc1_cross_layer = nn.Linear(in_features=round(planes / 2), out_features=round(planes / 8))
        self.fc2_cross_layer = nn.Linear(in_features=round(planes / 8), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prune_rate = 0.2


    def forward(self, x):
        residual = x
        out = self.conv1(x)      
        if out.size(1) == x.size(1):
            sep = x
            # Squeeze
            sep = self.globalAvgPool(sep)
            sep = sep.view(sep.size(0), -1)
            sep = self.fc1(sep)
            sep = self.relu(sep)
            sep = self.fc2(sep)
            sep = self.sigmoid(sep)
            sep_vec = sep.view(-1).cpu().detach().numpy()
            # print('sep_vec',sep_vec.shape)
            sep_sort = np.sort(sep_vec)
            sep_threshold = sep_sort[int(self.prune_rate*len(sep_sort))]
            # print('sep',sep)
            # print('sep_threshold_np',sep_threshold)
            sep = self.relu(sep-sep_threshold)
            sep = sep.view(sep.size(0), sep.size(1), 1, 1)
        else:
            sep = x
            # Squeeze
            sep = self.globalAvgPool1(sep)
            sep = sep.view(sep.size(0), -1)
            sep = self.fc1_cross_layer(sep)
            sep = self.relu(sep)
            sep = self.fc2_cross_layer(sep)
            sep = self.sigmoid(sep)
            sep_vec = sep.view(-1).cpu().detach().numpy()
            sep_sort = np.sort(sep_vec)
            sep_threshold = sep_sort[int(self.prune_rate*len(sep_sort))]
            sep = self.relu(sep-sep_threshold)
            sep = sep.view(sep.size(0), sep.size(1), 1, 1)

        # Excitation
        out = out * sep
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEP(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(SEP, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet20(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEP(BasicBlock, [3, 3, 3], **kwargs)
    return model


def se_resnet32(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEP(BasicBlock, [5, 5, 5], **kwargs)
    return model


def se_resnet44(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEP(BasicBlock, [7, 7, 7], **kwargs)
    return model


def se_resnet56(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEP(BasicBlock, [9, 9, 9], **kwargs)
    return model


def se_resnet110(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEP(BasicBlock, [18, 18, 18], **kwargs)
    return model