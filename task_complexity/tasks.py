import torch
import torch.nn as nn

import os
import os.path as osp
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


class Task(nn.Module):
    def __init__(self, n_conv):
        super(Task, self).__init__()
        self.layers = self._make_layers(n_conv)
        self.adjust_size = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.apply(_weights_init)

    def _make_layers(self, n_relu):
        layers = [nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)]
        for _ in range(n_relu):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.adjust_size(out)
        return out


def task(n_conv):
    return Task(n_conv=n_conv)


def save_tasks(n_conv_list, save_path='./tasks'):
    if not osp.exists(save_path):
        os.makedirs(save_path)
    for n_conv in n_conv_list:
        torch.save(task(n_conv).state_dict(), osp.join(save_path, f"task_{n_conv}.pth"))


if __name__ == '__main__':
    save_tasks([0, 2, 8, 26, 80]) # corresponding to tasks with 0/2/8/26/80 ReLUs, or 1/3/9/27/81 CONVs