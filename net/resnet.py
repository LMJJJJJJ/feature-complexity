import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', omit_last_relu=False):
        super(BasicBlock, self).__init__()
        self.omit_last_relu = omit_last_relu
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            padding = (planes - in_planes) // 2
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, padding, padding), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.omit_last_relu:
            return out
        out = F.relu(out)
        return out


class BasicBlockStudent(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockStudent, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            padding = (planes - in_planes) // 2
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, padding, padding), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, disentangle=True):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.disentangle = disentangle

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        if self.disentangle:
            self.layer3[-1].omit_last_relu = True

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.disentangle:
            return out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetDisentangler(nn.Module):
    def __init__(self, block, n):
        super(ResNetDisentangler, self).__init__()
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, n, stride=1)
        self.layer2 = self._make_layer(block, 256, n, stride=2)
        self.layer3 = self._make_layer(block, 512, n, stride=2)
        self.adjust_channel = nn.Conv2d(512, 64, 1, 1)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adjust_channel(out)
        return out


class StudentOutConv(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentOutConv, self).__init__()
        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            Flatten(),
            nn.Linear(64, num_classes)
        )
        self.apply(_weights_init)

    def forward(self, x):
        out = self.out_conv(x)
        return out


def resnetdisentangler(n):
    return ResNetDisentangler(BasicBlockStudent, n)

def resnet8(disentangle=True):
    return ResNet(BasicBlock, [1, 1, 1], disentangle=disentangle)

def resnet14(disentangle=True):
    return ResNet(BasicBlock, [2, 2, 2], disentangle=disentangle)

def resnet20(disentangle=True):
    return ResNet(BasicBlock, [3, 3, 3], disentangle=disentangle)

def resnet32(disentangle=True):
    return ResNet(BasicBlock, [5, 5, 5], disentangle=disentangle)

def resnet44(disentangle=True):
    return ResNet(BasicBlock, [7, 7, 7], disentangle=disentangle)
