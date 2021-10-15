PROJECT_ROOT = "./"
import os
import sys
sys.path.append(PROJECT_ROOT)
import os.path as osp
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tools import save_obj, load_obj, AverageValueMeter, update_lr


class Trainer(object):
    def __init__(self):
        self.model = None
        self.LR_LIST = None
        self.optimizer = None
        self.criterion = None
        self.GPU_ID = None
        self.train_loader = None
        self.test_loader = None
        self.plot_dic = None
        self.EPOCH = None
        self.PATH = None
        self.NUM_EPOCHS = None
        self.BATCH_SIZE = None

    def prepare(self):
        self._generate_plot_dic()
        self._prepare_dataset()
        self._prepare_model()

    def _prepare_dataset(self):
        return NotImplementedError

    def _prepare_model(self):
        return NotImplementedError

    def _generate_plot_dic(self):
        self.plot_dic = {
            "train_acc": [],
            "test_acc": [],
            "loss": []
        }

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(16, 12))
        # train & test accuracy
        plt.subplot(2, 2, 1)
        x = range(1, len(self.plot_dic["train_acc"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["train_acc"], label="train_acc")
        plt.plot(x, self.plot_dic["test_acc"], label="test_acc")
        plt.legend()
        # label loss (CrossEntropy) on training set
        plt.subplot(2, 2, 2)
        x = range(1, len(self.plot_dic["loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["loss"], label="loss")
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def train_model(self):
        self.model.train()
        step = 0
        train_loss = 0
        correct = 0
        print("Learning rate is", self.LR_LIST[self.EPOCH - 1])
        update_lr(self.optimizer, [self.LR_LIST[self.EPOCH - 1]])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.data
            y_pred = output.data.max(1)[1]

            correct += y_pred.eq(labels.data).sum()
            step += 1
            if step % 100 == 0:
                print(' Loss: {:.4f}'.format(loss.item()))

        acc = 100. * float(correct) / len(self.train_loader.dataset)
        length = len(self.train_loader.dataset) // self.BATCH_SIZE
        train_loss = train_loss / length
        self.plot_dic["train_acc"].append(acc)
        self.plot_dic["loss"].append(train_loss.item())
        return train_loss, acc

    def eval_model(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
                output = self.model(images)
                prediction = output.data.max(1)[1]
                correct += prediction.eq(labels.data).sum()
        acc = 100. * float(correct) / len(self.test_loader.dataset)
        self.plot_dic["test_acc"].append(acc)
        print('Accuracy of the network on the test images: {} %'.format(acc))
        return acc

    def save(self):
        print("Saving...")
        torch.save(self.model.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])
        
    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            train_loss, train_acc = self.train_model()
            print("Train Set: Loss: {0:2.3f} Acc: {1:.3f}%".format(train_loss, train_acc))
            acc = self.eval_model()
            self.draw()
            self.save()
        torch.save(self.model.to("cpu").state_dict(), self.PATH['model_save_path'])


class Trainer_CIFAR10(Trainer):
    def __init__(self, args):
        super(Trainer_CIFAR10, self).__init__()
        self.BATCH_SIZE = args.batch_size
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.NUM_EPOCHS = args.num_epochs
        self.MODEL_NAME = args.model_name
        if args.data_size is None:
            args.data_size = 'strong'
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = f'./data/CIFAR10/CIFAR10{args.data_size}'
        if args.data_size == 'strong':
            self.DATA_ROOT = './data/CIFAR10'

        self.PATH = {}
        self.PATH["save_root"]           = osp.join(PROJECT_ROOT,
                                                    f"pretrained_models/{self.MODEL_NAME}_cifar10")
        self.PATH['model_save_path']     = osp.join(self.PATH["save_root"],
                                                    f"{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        self.PATH['fig_save_path']       = osp.join(self.PATH["save_root"],
                                                    f"{self.MODEL_NAME}_{self.DATA_SIZE}_curve.png")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["save_root"],
                                                    f"{self.MODEL_NAME}_{self.DATA_SIZE}_data.bin")

        if not os.path.exists(self.PATH["save_root"]):
            os.makedirs(self.PATH["save_root"])

    def _prepare_dataset(self):
        print(self.DATA_ROOT)
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.DATA_SIZE == 200:
            from dataset.cifar10 import CIFAR10200
            train_set = CIFAR10200(self.DATA_ROOT, train=True, transform=transform_train)
            test_set = CIFAR10200(self.DATA_ROOT, train=False, transform=transform_test)
        elif self.DATA_SIZE == 500:
            from dataset.cifar10 import CIFAR10500
            train_set = CIFAR10500(self.DATA_ROOT, train=True, transform=transform_train)
            test_set = CIFAR10500(self.DATA_ROOT, train=False, transform=transform_test)
        elif self.DATA_SIZE == 1000:
            from dataset.cifar10 import CIFAR101000
            train_set = CIFAR101000(self.DATA_ROOT, train=True, transform=transform_train)
            test_set = CIFAR101000(self.DATA_ROOT, train=False, transform=transform_test)
        elif self.DATA_SIZE == 2000:
            from dataset.cifar10 import CIFAR102000
            train_set = CIFAR102000(self.DATA_ROOT, train=True, transform=transform_train)
            test_set = CIFAR102000(self.DATA_ROOT, train=False, transform=transform_test)
        elif self.DATA_SIZE == 5000:
            from dataset.cifar10 import CIFAR105000
            train_set = CIFAR105000(self.DATA_ROOT, train=True, transform=transform_train)
            test_set = CIFAR105000(self.DATA_ROOT, train=False, transform=transform_test)
        elif self.DATA_SIZE == 'strong':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            train_set = datasets.CIFAR10(self.DATA_ROOT, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(self.DATA_ROOT, train=False, transform=transform_test)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=1)

    def _prepare_model(self):
        if self.MODEL_NAME == 'resnet8':
            from net.resnet import resnet8
            self.model = resnet8(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet14':
            from net.resnet import resnet14
            self.model = resnet14(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet20':
            from net.resnet import resnet20
            self.model = resnet20(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet32':
            from net.resnet import resnet32
            self.model = resnet32(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet44':
            from net.resnet import resnet44
            self.model = resnet44(disentangle=False).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.CrossEntropyLoss()


class Trainer_CUB(Trainer):
    def __init__(self, args):
        super(Trainer_CUB, self).__init__()
        self.BATCH_SIZE = args.batch_size
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.NUM_EPOCHS = args.num_epochs
        if args.data_size is None:
            args.data_size = 'strong'
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = f"./data/CUB/CUB_{args.data_size}"
        self.MODEL_NAME = args.model_name

        self.PATH = {}
        self.PATH["save_root"] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_cub")
        self.PATH['model_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        self.PATH['fig_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_curve.png")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_data.bin")

        if not os.path.exists(self.PATH["save_root"]):
            os.makedirs(self.PATH["save_root"])

    def _prepare_dataset(self):
        print(self.DATA_ROOT)
        CUB_mean = [0.485, 0.456, 0.406]
        CUB_std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        train_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "train"), train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_model(self):
        if self.MODEL_NAME == "resnet18":
            from net.resnetbig import resnet18_cub
            self.model = resnet18_cub(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet34":
            from net.resnetbig import resnet34_cub
            self.model = resnet34_cub(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == "vgg16":
            from net.vggbig import vgg16_cub
            self.model = vgg16_cub(disentangle=False).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.CrossEntropyLoss()


class Trainer_DOGS(Trainer):
    def __init__(self, args):
        super(Trainer_DOGS, self).__init__()
        self.BATCH_SIZE = args.batch_size
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.NUM_EPOCHS = args.num_epochs
        if args.data_size is None:
            args.data_size = 'strong'
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = f"./data/DOGS/DOGS_{args.data_size}"
        self.MODEL_NAME = args.model_name

        self.PATH = {}
        self.PATH["save_root"] = osp.join(PROJECT_ROOT, "pretrained_models/{}_dogs".format(self.MODEL_NAME))
        self.PATH['model_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        self.PATH['fig_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_curve.png")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["save_root"], f"{self.MODEL_NAME}_{self.DATA_SIZE}_data.bin")

        if not os.path.exists(self.PATH["save_root"]):
            os.makedirs(self.PATH["save_root"])

    def _prepare_dataset(self):
        print(self.DATA_ROOT)
        DOGS_mean = [0.485, 0.456, 0.406]
        DOGS_std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DOGS_mean, DOGS_std)
        ])
        train_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "train"), train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DOGS_mean, DOGS_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_model(self):
        if self.MODEL_NAME == "resnet18":
            from net.resnetbig import resnet18_dogs
            self.model = resnet18_dogs(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet34":
            from net.resnetbig import resnet34_dogs
            self.model = resnet34_dogs(disentangle=False).to(self.GPU_ID)
        elif self.MODEL_NAME == "vgg16":
            from net.vggbig import vgg16_dogs
            self.model = vgg16_dogs(disentangle=False).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.CrossEntropyLoss()


def main():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--model-name', type=str, default=None, help='model to train')
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None, help='dataset: cifar10, cub, dogs')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu')
    parser.add_argument('--initial-lr', type=int, default=-3)
    parser.add_argument('--final-lr', type=int, default=-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=300)
    args = parser.parse_args()
    print(args)

    if args.dataset == 'cifar10':
        trainer = Trainer_CIFAR10(args)
        trainer.prepare()
        trainer.run()
    elif args.dataset == 'dogs':
        trainer = Trainer_DOGS(args)
        trainer.prepare()
        trainer.run()
    elif args.dataset == 'cub':
        trainer = Trainer_CUB(args)
        trainer.prepare()
        trainer.run()


if __name__ == '__main__':
    main()