import os
import os.path as osp
PROJECT_ROOT = osp.join("./")
import sys
sys.path.append(PROJECT_ROOT)
sys.path.append("../")

import argparse

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from net.resnet import resnetdisentangler
from tools import save_obj, load_obj, update_lr




class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def get_current(cur_block_num, previous_path):
    current = resnetdisentangler(cur_block_num)
    current_dict = current.state_dict()
    previous_dict = torch.load(previous_path)
    load_dict = {k: v for k, v in previous_dict.items() if k in current_dict}
    # for k, v in load_dict.items():
    #     print(k)
    current_dict.update(load_dict)
    current.load_state_dict(current_dict)
    return current


class DistillationTrainer(object):
    def __init__(self, args):
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.NUM_EPOCHS = args.num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.TASK_DIFFICULTY = args.task_difficulty
        self.DATA_ROOT = "./data"
        self.EPOCH = 1
        self.PATH = {}
        self.PATH["fig_save_folder"] = osp.join(PROJECT_ROOT, f"fig/task_{self.TASK_DIFFICULTY}")
        self.PATH['fig_save_path'] = osp.join(self.PATH["fig_save_folder"], "{}_curve.png".format(self.block_num))
        self.PATH["model_save_folder"] = osp.join(PROJECT_ROOT, f"models/task_{self.TASK_DIFFICULTY}")
        self.PATH['model_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.block_num))
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_data.bin".format(self.block_num))
        self.PATH['pretrained_net'] = osp.join(PROJECT_ROOT, f"pretrained_models/task_{self.TASK_DIFFICULTY}/task_{self.TASK_DIFFICULTY}_model.pth")
        if len(self.PREVIOUS_LIST) == 0:
            self.PATH['previous_model'] = None
        else:
            self.PATH['previous_model'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

        if not os.path.exists(self.PATH["fig_save_folder"]):
            os.makedirs(self.PATH["fig_save_folder"])
        if not os.path.exists(self.PATH["model_save_folder"]):
            os.makedirs(self.PATH["model_save_folder"])

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()
        self._generate_plot_dic()

    def _generate_plot_dic(self):
        self.plot_dic = {
            "train_loss": [],
            "test_loss": []
        }

    def _prepare_dataset(self):
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
        dataset_train = datasets.CIFAR10(self.DATA_ROOT, train=True, download=True, transform=transform_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.BATCH_SIZE,
            sampler=sampler_train,
            num_workers=1
        )
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.DATA_ROOT, train=False, transform=transform_test),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    def _prepare_model(self):
        self.pretrained_model = resnetdisentangler(6).to(self.GPU_ID)
        self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
        print(f"Pretrained Teacher {self.PATH['pretrained_net']}")

        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentangler(self.block_num).to(self.GPU_ID)
        else:
            self.current = get_current(self.block_num, self.PATH['previous_model']).to(self.GPU_ID)
            print("{} loaded".format(self.PATH['previous_model']))
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.LR_LIST[0], betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()

    def train_epoch(self):
        self.current.train()
        self.pretrained_model.eval()
        train_loss = 0
        update_lr(self.optimizer, [self.LR_LIST[self.EPOCH - 1]])
        print("Learning rate is", self.optimizer.param_groups[0]["lr"])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images = images.to(self.GPU_ID)
            self.optimizer.zero_grad()
            target = self.pretrained_model(images).detach()
            output = self.current(images)
            loss = self.criterion(output, target)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.data * images.shape[0]

        train_loss = train_loss / len(self.train_loader.sampler)
        self.plot_dic['train_loss'].append(train_loss.item())
        print("Train loss: {0:.5f}".format(train_loss))

    def eval_epoch(self):
        self.current.eval()
        self.pretrained_model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images = images.to(self.GPU_ID)
                target = self.pretrained_model(images).detach()
                output = self.current(images)
                test_loss += self.criterion(output, target) * images.shape[0]

        test_loss = test_loss / len(self.test_loader.dataset)
        self.plot_dic['test_loss'].append(test_loss)
        print("Test loss: {0:.5f}".format(test_loss))

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            self.train_epoch()
            # self.eval_epoch()
            self.save()
            self.draw()
        torch.save(self.current.to("cpu").state_dict(), self.PATH['model_save_path'])

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(16, 12))
        # train & test accuracy
        plt.subplot(2, 2, 1)
        x = range(1, len(self.plot_dic["train_loss"]) + 1)
        plt.xlabel("epoch")
        plt.ylim([0, 1])
        plt.plot(x, self.plot_dic["train_loss"], label="train_loss")
        plt.legend()
        # label loss (CrossEntropy) on training set
        plt.subplot(2, 2, 2)
        x = range(1, len(self.plot_dic["test_loss"]) + 1)
        plt.xlabel("epoch")
        plt.ylim([0, 1])
        plt.plot(x, self.plot_dic["test_loss"], label="test_loss")
        plt.legend()
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        torch.save(self.current.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


if __name__ == '__main__':
    train_list = [1, 2, 4, 8, 16, 32]
    bs_list = [128, 128, 128, 64, 64, 32]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument("--initial-lr", type=int, default=-3)
    parser.add_argument("--final-lr", type=int, default=-5)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--block-num', type=int, default=None)
    parser.add_argument('--previous-list', type=list, default=None)
    parser.add_argument('--task-difficulty', type=int, default=None)
    args = parser.parse_args()
    print(args)
    for i in range(0, len(train_list)):
        args.block_num = train_list[i]
        args.previous_list = train_list[:i]
        args.batch_size = bs_list[i]
        print(args)
        d = DistillationTrainer(args)
        d.prepare()
        d.run()
