import os
import os.path as osp
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms


class CIFAR10200(data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        filename = "0%05d" % (index + 1) if self.train else "1%05d" % (index + 1)
        if self.train:
            image = torch.load(osp.join(self.root, "train", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))
        else:
            image = torch.load(osp.join(self.root, "test", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.train:
            return 200
        else:
            return 10000


class CIFAR10500(data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        filename = "0%05d" % (index + 1) if self.train else "1%05d" % (index + 1)
        if self.train:
            image = torch.load(osp.join(self.root, "train", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))
        else:
            image = torch.load(osp.join(self.root, "test", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.train:
            return 500
        else:
            return 10000


class CIFAR101000(data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        filename = "0%05d" % (index + 1) if self.train else "1%05d" % (index + 1)
        if self.train:
            image = torch.load(osp.join(self.root, "train", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))
        else:
            image = torch.load(osp.join(self.root, "test", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.train:
            return 1000
        else:
            return 10000


class CIFAR102000(data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        filename = "0%05d" % (index + 1) if self.train else "1%05d" % (index + 1)
        if self.train:
            image = torch.load(osp.join(self.root, "train", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))
        else:
            image = torch.load(osp.join(self.root, "test", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.train:
            return 2000
        else:
            return 10000


class CIFAR105000(data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        filename = "0%05d" % (index + 1) if self.train else "1%05d" % (index + 1)
        if self.train:
            image = torch.load(osp.join(self.root, "train", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))
        else:
            image = torch.load(osp.join(self.root, "test", "%s.pt" % filename))
            label = torch.load(osp.join(self.root, "label", "%s.pt" % filename))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.train:
            return 5000
        else:
            return 10000

