PROJECT_ROOT = "./"
import sys
sys.path.append(PROJECT_ROOT)

import argparse
import os.path as osp
import os
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

from vgg import vgg16_bn
from resnetbig import resnetstudent
from tools.lib import save_obj, load_obj, update_lr

GPU_ID = 3

EPOCH_NUM = 500


MODEL = "vgg16bn"
print(MODEL)

LR_LIST = np.logspace(-3, -5, EPOCH_NUM)
FOLDER = 1


DATA_ROOT = "../data/CUB/CUB_strong"
CUB_mean = [0.485, 0.456, 0.406]
CUB_std = [0.229, 0.224, 0.225]


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def get_current(cur_block_num, previous_path):
    current = resnetstudent(cur_block_num)
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
        self.LEARNING_RATE = args.learning_rate
        self.NUM_EPOCHS = args.num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.data_root = DATA_ROOT
        self.EPOCH = 1
        self.PATH = {}
        self.PATH["fig_save_folder"] = osp.join(PROJECT_ROOT, "fig_cub/{}/{}/{}".format(MODEL, MODEL, FOLDER))
        self.PATH['fig_save_path'] = osp.join(self.PATH["fig_save_folder"], "{}_steplr.png".format(self.block_num))
        self.PATH["model_save_folder"] = osp.join(PROJECT_ROOT, "models")
        self.PATH['model_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_steplr.pth".format(self.block_num))
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_steplr_data.bin".format(self.block_num))
        if len(self.PREVIOUS_LIST) == 0:
            self.PATH['previous_student'] = None
        else:
            self.PATH['previous_student'] = osp.join(self.PATH["model_save_folder"], "{}_steplr.pth".format(self.PREVIOUS_LIST[-1]))

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
            "distill_loss": []
        }

    def _prepare_dataset(self):
        print(self.data_root)

        train_transform = transforms.Compose([
            transforms.RandomCrop(224, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        train_set = datasets.ImageFolder(osp.join(self.data_root, "train"), train_transform)
        sampler_train = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, sampler=sampler_train)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.data_root, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_model(self):
        self.pretrained_model = vgg16_bn(pretrained=True).to(self.GPU_ID)

        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetstudent(self.block_num).to(self.GPU_ID)
        else:
            self.current = get_current(self.block_num, self.PATH['previous_student']).to(self.GPU_ID)
            print("{} loaded".format(self.PATH['previous_student']))
        # self.optimizer = torch.optim.SGD(self.current.parameters(), lr=self.LEARNING_RATE, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.LEARNING_RATE, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)

    def get_pretrained(self, x):
        return self.pretrained_model.extract_feature(x, 4).detach()

    def train_epoch(self):
        self.current.train()
        # self.in_conv.eval()
        self.pretrained_model.eval()
        step = 0
        distill_loss = 0
        update_lr(self.optimizer, [LR_LIST[self.EPOCH - 1]])
        print("Learning rate is", self.optimizer.param_groups[0]["lr"])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
            self.optimizer.zero_grad()
            pretrained = self.get_pretrained(images).detach()
            output = self.current(images)

            # 优化除了outconv的部分
            loss = self.criterion(output, pretrained)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            distill_loss += loss.data

            step += 1
            if step % 100 == 0:
                print(' distillation loss: {:.4f}'.format(loss.item()))

        length = len(self.train_loader.sampler) // self.BATCH_SIZE # revised
        distill_loss = distill_loss / length

        self.plot_dic['distill_loss'].append(distill_loss.item())

        print("Train Set: distillation Loss (feature): {0:2.3f}".format(distill_loss))

    def eval_epoch(self):
        self.current.eval()
        # self.out_conv_student.eval()
        self.pretrained_model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                output = self.current(images)


        length = len(self.test_loader.dataset) // self.BATCH_SIZE
        # return distill_acc, pretrained_acc

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
        # distillation loss (MSE) on training set
        x = range(1, len(self.plot_dic["distill_loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["distill_loss"], label="distill_loss")
        plt.legend()
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        torch.save(self.current.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


def distill():
    # train_list = [1, 4, 7, 10, 13, 15]
    train_list = [1, 2, 4, 8, 16, 32]
    bs_list = [32, 32, 32, 32, 16, 8]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--learning-rate', type=float, default=LR_LIST[0], help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-id', type=int, default=GPU_ID)
    parser.add_argument('--num-epochs', type=int, default=EPOCH_NUM) # big enough to converge????
    parser.add_argument('--model-num', type=int, default=1, help='model id')
    parser.add_argument('--block-num', type=int, default=None, help='train with n blocks')
    parser.add_argument('--previous-list', type=list, default=None, help='previous models')
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


if __name__ == '__main__':
    distill()
