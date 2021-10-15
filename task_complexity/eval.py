import os
import os.path as osp

PROJECT_ROOT = "./"
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
from tools import save_obj, load_obj, update_lr, AverageValueMeter

FEATURE_DIM = 64 * 8 * 8

DATA_DICT_PATH = None
DATA_DICT = None


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Evaluator(object):
    def __init__(self, args):
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.PREVIOUS_LIST = args.previous_list
        self.TASK_DIFFICULTY = args.task_difficulty
        self.MODEL_ROOT = osp.join(PROJECT_ROOT, f"models/task_{self.TASK_DIFFICULTY}")
        self.EPOCH = 1
        self.DATA_ROOT = "./data"
        self.PATH = {}
        self.PATH['target_model'] = osp.join(PROJECT_ROOT, f"pretrained_models/task_{self.TASK_DIFFICULTY}/task_{self.TASK_DIFFICULTY}_model.pth")
        self.PATH['disentangled_model'] = osp.join(self.MODEL_ROOT, "{}_model.pth".format(self.block_num))
        if len(self.PREVIOUS_LIST) != 0:
            self.PATH['disentangled_previous'] = osp.join(self.MODEL_ROOT, "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

        self.prepare()

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()

    def _prepare_dataset(self):
        print(self.DATA_ROOT)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.DATA_ROOT, train=True, download=True, transform=transform),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.DATA_ROOT, train=False, transform=transform),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    def _prepare_model(self):
        self.pretrained_model = resnetdisentangler(6).to(self.GPU_ID)
        self.pretrained_model.load_state_dict(torch.load(self.PATH['target_model']))
        print(f"Pretrained Student {self.PATH['target_model']}")

        self.current = resnetdisentangler(self.block_num).to(self.GPU_ID)
        self.current.load_state_dict(torch.load(self.PATH['disentangled_model']))

        if len(self.PREVIOUS_LIST) != 0:
            self.previous = resnetdisentangler(self.PREVIOUS_LIST[-1]).to(self.GPU_ID)
            self.previous.load_state_dict(torch.load(self.PATH['disentangled_previous']))

        self.criterion = nn.MSELoss()

    def add_data(self, mode, data_type, value):
        DATA_DICT[mode][data_type][self.block_num] = value
        print(f" - {mode} {data_type}: {value}")

    def _calculate_loss(self, criterion, pred, target):
        loss = 0
        assert pred.shape[0] == target.shape[0]
        for i in range(pred.shape[0]):
            loss += criterion(pred[i].unsqueeze(0), target[i].unsqueeze(0)).data
        return loss


    def eval_train(self):
        if len(self.PREVIOUS_LIST) != 0:
            self.previous.eval()
        self.current.eval()
        self.pretrained_model.eval()

        meter_teacher = AverageValueMeter(gpu_id=self.GPU_ID, dim=FEATURE_DIM)
        meter_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=FEATURE_DIM)
        meter_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=FEATURE_DIM)
        meter_remain = AverageValueMeter(gpu_id=self.GPU_ID, dim=FEATURE_DIM)

        train_distill_loss = 0

        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc="Train set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                f_current = self.current(images).detach()
                f_teacher = self.pretrained_model(images).detach()
                train_distill_loss += self._calculate_loss(self.criterion, f_current, f_teacher)
                f_remain = f_teacher - f_current
                if len(self.PREVIOUS_LIST) != 0:
                    f_previous = self.previous(images)
                    f_new = f_current - f_previous
                else:
                    f_new = f_current

                meter_teacher.add(f_teacher)
                meter_current.add(f_current)
                meter_new.add(f_new)
                meter_remain.add(f_remain)

        train_distill_loss = train_distill_loss / len(self.train_loader.dataset)
        std_teacher = meter_teacher.get_std().item()
        std_current = meter_current.get_std().item()
        std_new = meter_new.get_std().item()
        std_remain = meter_remain.get_std().item()
        self.add_data("train", "train_distill_loss", train_distill_loss.item())
        self.add_data("train", "std_teacher", std_teacher)
        self.add_data("train", "std_current", std_current)
        self.add_data("train", "std_new", std_new)
        self.add_data("train", "std_remain", std_remain)

    def eval_test(self):
        self.current.eval()
        self.pretrained_model.eval()
        test_distill_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Test set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
                f_current = self.current(images).detach()
                f_teacher = self.pretrained_model(images).detach()
                test_distill_loss += self._calculate_loss(self.criterion, f_current, f_teacher)

        test_distill_loss = test_distill_loss / len(self.test_loader.dataset)
        self.add_data("test", "test_distill_loss", test_distill_loss.item())

    def run(self):
        self.eval_train()
        self.eval_test()


def create_data_dict(task_difficulty):
    task_difficulty = f"task_{task_difficulty}"
    student_depth = [1, 2, 4, 8, 16, 32]
    train_data = ["train_distill_loss", "std_teacher", "std_current", "std_new", "std_remain"]
    test_data = ["test_distill_loss"]
    data_dict = {
        "train": {data: {depth: None for depth in student_depth} for data in train_data},
        "test": {data: {depth: None for depth in student_depth} for data in test_data}
    }
    save_obj(data_dict, f"./data/{task_difficulty}_data.bin")


if __name__ == '__main__':
    train_list = [1, 2, 4, 8, 16, 32]
    bs_list = [128, 128, 128, 128, 128, 128]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--block-num', type=int, default=None)
    parser.add_argument('--previous-list', type=list, default=None)
    parser.add_argument('--task-difficulty', type=int, default=None)
    args = parser.parse_args()
    create_data_dict(args.task_difficulty)
    DATA_DICT_PATH = osp.join(PROJECT_ROOT, f"data/task_{args.task_difficulty}_data.bin")
    DATA_DICT = load_obj(DATA_DICT_PATH)
    print(args)
    for i in range(0, len(train_list)):
        args.block_num = train_list[i]
        args.previous_list = train_list[:i]
        args.batch_size = bs_list[i]
        print()
        print(args)
        d = Evaluator(args)
        d.run()
        save_obj(DATA_DICT, DATA_DICT_PATH)




