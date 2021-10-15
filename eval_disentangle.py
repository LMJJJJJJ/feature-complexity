import os
import os.path as osp
PROJECT_ROOT = "./"
import sys
sys.path.append(PROJECT_ROOT)

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

from tools import save_obj, load_obj, update_lr, AverageValueMeter
from net.resnet import resnetdisentangler
from net.resnetbig import resnetdisentanglerbig

DATA_DICT_PATH = None
DATA_DICT = None



def create_data_dict(dataset, model_name, data_size):
    if not osp.exists(f"./data/disentangle_{dataset}"):
        os.makedirs(f"./data/disentangle_{dataset}")
    depths = [1, 2, 4, 8, 16, 32]
    train_data = ["train_teacher_acc", "train_teacher_lbl_loss", "train_student_acc", "train_student_lbl_loss",
                  "distill_loss",
                  "std_teacher", "std_current", "std_new", "std_remain"]
    test_data = ["test_teacher_acc", "test_teacher_lbl_loss", "test_student_acc", "test_student_lbl_loss"]
    data_dict = {
        "train": {data: {depth: None for depth in depths} for data in train_data},
        "test": {data: {depth: None for depth in depths} for data in test_data}
    }
    save_obj(data_dict, f"./data/disentangle_{dataset}/{model_name}_{data_size}_data.bin")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Evaluator(object):
    def __init__(self):
        self.DATA_ROOT = None
        self.GPU_ID = None
        self.block_num = None
        self.BATCH_SIZE = None
        self.PREVIOUS_LIST = None
        self.FEATURE_DIM = None
        self.EPOCH = 1
        self.PATH = {}

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.train_loader = None
        self.test_loader = None
        raise NotImplementedError

    def _prepare_model(self):
        self.pretrained_model = None
        self.previous = None
        self.current = None
        self.out_conv = None
        self.criterion = None
        self.label_criterion = None
        raise NotImplementedError

    def add_data(self, mode, data_type, value):
        DATA_DICT[mode][data_type][self.block_num] = value
        print(f" - {mode} {data_type}: {value}")

    def calculate_loss(self, criterion, prediction, target):
        assert prediction.shape[0] == target.shape[0]
        loss = 0
        for i in range(prediction.shape[0]):
            loss += criterion(prediction[i].unsqueeze(0), target[i].unsqueeze(0)).data
        return loss

    def eval_train(self):
        if len(self.PREVIOUS_LIST) != 0:
            self.previous.eval()
        self.current.eval()
        self.out_conv.eval()
        self.pretrained_model.eval()

        meter_teacher = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_remain = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)

        correct_student = 0
        correct_teacher = 0
        lbl_loss_student = 0
        lbl_loss_teacher = 0
        distill_loss = 0


        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc="Train set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                f_current = self.current(images)
                f_teacher = self.pretrained_model(images).detach()
                distill_loss += self.calculate_loss(self.criterion, f_current, f_teacher)

                pred_student = self.out_conv(f_current)
                correct_student += pred_student.data.max(1)[1].eq(labels.data).sum()
                lbl_loss_student += self.calculate_loss(self.label_criterion, pred_student, labels)

                pred_teacher = self.out_conv(f_teacher)
                correct_teacher += pred_teacher.data.max(1)[1].eq(labels.data).sum()
                lbl_loss_teacher += self.calculate_loss(self.label_criterion, pred_teacher, labels)

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

        acc_student = 100. * float(correct_student) / len(self.train_loader.dataset)
        acc_teacher = 100. * float(correct_teacher) / len(self.train_loader.dataset)
        lbl_loss_student = lbl_loss_student / len(self.train_loader.dataset)
        lbl_loss_teacher = lbl_loss_teacher / len(self.train_loader.dataset)
        distill_loss = distill_loss / len(self.train_loader.dataset)
        std_teacher = meter_teacher.get_std().item()
        std_current = meter_current.get_std().item()
        std_new = meter_new.get_std().item()
        std_remain = meter_remain.get_std().item()

        self.add_data("train", "train_teacher_acc", acc_teacher)
        self.add_data("train", "train_teacher_lbl_loss", lbl_loss_teacher.item())
        self.add_data("train", "train_student_acc", acc_student)
        self.add_data("train", "train_student_lbl_loss", lbl_loss_student.item())
        self.add_data("train", "distill_loss", distill_loss.item())
        self.add_data("train", "std_teacher", std_teacher)
        self.add_data("train", "std_current", std_current)
        self.add_data("train", "std_new", std_new)
        self.add_data("train", "std_remain", std_remain)

    def eval_test(self):
        self.current.eval()
        self.out_conv.eval()
        self.pretrained_model.eval()
        correct_student = 0
        correct_teacher = 0
        lbl_loss_student = 0
        lbl_loss_teacher = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Test set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                pred_student = self.out_conv(self.current(images))
                correct_student += pred_student.data.max(1)[1].eq(labels.data).sum()
                lbl_loss_student += self.calculate_loss(self.label_criterion, pred_student, labels)

                pred_teacher = self.out_conv(self.pretrained_model(images))
                correct_teacher += pred_teacher.data.max(1)[1].eq(labels.data).sum()
                lbl_loss_teacher += self.calculate_loss(self.label_criterion, pred_teacher, labels)

        acc_student = 100. * float(correct_student) / len(self.test_loader.dataset)
        acc_teacher = 100. * float(correct_teacher) / len(self.test_loader.dataset)
        lbl_loss_student = lbl_loss_student / len(self.test_loader.dataset)
        lbl_loss_teacher = lbl_loss_teacher / len(self.test_loader.dataset)
        self.add_data("test", "test_teacher_acc", acc_teacher)
        self.add_data("test", "test_teacher_lbl_loss", lbl_loss_teacher.item())
        self.add_data("test", "test_student_acc", acc_student)
        self.add_data("test", "test_student_lbl_loss", lbl_loss_student.item())

    def run(self):
        self.eval_train()
        self.eval_test()


class Eval_CIFAR10(Evaluator):
    def __init__(self, args):
        super(Eval_CIFAR10, self).__init__()
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.FEATURE_DIM = args.feature_dim
        self.MODEL_SIZE = f"{args.model_name}_{args.data_size}"
        self.DATA_ROOT = f"./data/CIFAR10/CIFAR10{args.data_size}"
        self.MODEL_ROOT = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "disentangle")

        self.PATH = {}
        self.PATH['pretrained_model'] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_cifar10/{self.MODEL_SIZE}_model.pth")
        self.PATH['pretrained_disentangler'] = osp.join(self.MODEL_ROOT, "{}_model.pth".format(self.block_num))
        if len(self.PREVIOUS_LIST) != 0:
            self.PATH['pretrained_previous'] = osp.join(self.MODEL_ROOT, "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

    def _prepare_dataset(self):
        if self.DATA_SIZE == 200:
            from dataset.cifar10 import CIFAR10200
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.train_loader = torch.utils.data.DataLoader(
                CIFAR10200(self.DATA_ROOT, train=True, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
            self.test_loader = torch.utils.data.DataLoader(
                CIFAR10200(self.DATA_ROOT, train=False, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        elif self.DATA_SIZE == 500:
            from dataset.cifar10 import CIFAR10500
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.train_loader = torch.utils.data.DataLoader(
                CIFAR10500(self.DATA_ROOT, train=True, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
            self.test_loader = torch.utils.data.DataLoader(
                CIFAR10500(self.DATA_ROOT, train=False, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        elif self.DATA_SIZE == 1000:
            from dataset.cifar10 import CIFAR101000
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.train_loader = torch.utils.data.DataLoader(
                CIFAR101000(self.DATA_ROOT, train=True, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
            self.test_loader = torch.utils.data.DataLoader(
                CIFAR101000(self.DATA_ROOT, train=False, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        elif self.DATA_SIZE == 2000:
            from dataset.cifar10 import CIFAR102000
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.train_loader = torch.utils.data.DataLoader(
                CIFAR102000(self.DATA_ROOT, train=True, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
            self.test_loader = torch.utils.data.DataLoader(
                CIFAR102000(self.DATA_ROOT, train=False, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        elif self.DATA_SIZE == 5000:
            from dataset.cifar10 import CIFAR105000
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.train_loader = torch.utils.data.DataLoader(
                CIFAR105000(self.DATA_ROOT, train=True, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
            self.test_loader = torch.utils.data.DataLoader(
                CIFAR105000(self.DATA_ROOT, train=False, transform=transform),
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        else:
            raise Exception("Failed in loading dataset.")

    def _prepare_model(self):
        if self.MODEL_NAME == "resnet8":
            from net.resnet import resnet8
            self.pretrained_model = resnet8(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet14":
            from net.resnet import resnet14
            self.pretrained_model = resnet14(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet20":
            from net.resnet import resnet20
            self.pretrained_model = resnet20(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet32":
            from net.resnet import resnet32
            self.pretrained_model = resnet32(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet44":
            from net.resnet import resnet44
            self.pretrained_model = resnet44(disentangle=True).to(self.GPU_ID)
        self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_model']))

        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            Flatten(),
            self.pretrained_model.linear
        )

        self.current = resnetdisentangler(self.block_num).to(self.GPU_ID)
        self.current.load_state_dict(torch.load(self.PATH['pretrained_disentangler']))

        if len(self.PREVIOUS_LIST) != 0:
            self.previous = resnetdisentangler(self.PREVIOUS_LIST[-1]).to(self.GPU_ID)
            self.previous.load_state_dict(torch.load(self.PATH['pretrained_previous']))

        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


if __name__ == '__main__':
    train_list = [1, 2, 4, 8, 16, 32]
    bs_list = [128, 128, 128, 128, 128, 128]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--block-num', type=int, default=None, help='train with n blocks')
    parser.add_argument('--previous-list', type=list, default=None, help='previous models')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--feature-dim', type=int, default=None)
    args = parser.parse_args()
    print(args)
    create_data_dict(args.dataset, args.model_name, args.data_size)
    if args.dataset == "cifar10":
        args.feature_dim = 8 * 8 * 64
        DATA_DICT_PATH = f"./data/disentangle_{args.dataset}/{args.model_name}_{args.data_size}_data.bin"
        DATA_DICT = load_obj(DATA_DICT_PATH)
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = bs_list[i]
            print(args)
            evaluator = Eval_CIFAR10(args)
            evaluator.prepare()
            evaluator.run()
            save_obj(DATA_DICT, DATA_DICT_PATH)