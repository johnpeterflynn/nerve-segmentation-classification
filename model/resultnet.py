import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

from base import BaseModel


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResUltNet(BaseModel):
    """
    A PyTorch implementation of ResUltNet

    """
    def __init__(self, params, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=3):
        self.inplanes = 64
        super(ResUltNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        #self.conv1.requires_grad = False
        #self.bn1.requires_grad = False
        #self.relu.requires_grad = False
        #self.maxpool.requires_grad = False
        #self.layer1.requires_grad = False
        #self.layer2.requires_grad = False
        #self.layer3.requires_grad = False


        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        #for param in self.parameters():
        #    param.requires_grad = False

        self.fc = nn.Linear(25088 * block.expansion, num_classes) # 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self._load_pretrained_resnet()

    def _load_pretrained_resnet(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = self.state_dict()
        #pretrained_dict = [(k, v) for k, v in pretrained_dict.items()]
        #print(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (model_dict[k].shape == pretrained_dict[k].shape)}
        #for k, v in pretrained_dict.items():
        #    print('pretraied keys, size: ', k, ', ', len(v))
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

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
        #print('size 1: ', x.shape)
        x = self.conv1(x)
        #print('size 2: ', x.shape)
        x = self.bn1(x)
        #print('size 3: ', x.shape)
        x = self.relu(x)
        #print('size 4: ', x.shape)
        x = self.maxpool(x)
        #print('size 5: ', x.shape)
        x = self.layer1(x)
        #print('size 6: ', x.shape)
        x = self.layer2(x)
        #print('size 7: ', x.shape)
        x = self.layer3(x)
        #print('size 8: ', x.shape)
        x = self.layer4(x)
        #print('size 9: ', x.shape)

        x = self.avgpool(x)
        #print('size 10: ', x.shape)
        x = x.view(x.size(0), -1)
        #print('size 11: ', x.shape)
        x = self.fc(x)
        #print('size 12: ', x.shape)

        return x
