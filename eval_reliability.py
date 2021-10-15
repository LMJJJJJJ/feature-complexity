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

from tools import save_obj, load_obj, update_lr, AverageValueMeter
from net.resnet import resnetdisentangler
from net.resnetbig import resnetdisentanglerbig

DATA_DICT_PATH = None
DATA_DICT = None


def create_data_dict(dataset, model_name, data_size):
    if not osp.exists(f"./data/reliability_{dataset}"):
        os.makedirs(f"./data/reliability_{dataset}")
    depths = [1, 2, 4, 8, 16, 32]
    train_data = ["var_teacher", "var_normal_current", "var_normal_new", "var_normal_remain",
                  "var_f_current", "var_f_new",
                  "var_af_current", "var_af_new",
                  "var_gaf_current", "var_gaf_new",
                  "var_agaf_current", "var_agaf_new",
                  "var_normal-agaf_current", "var_normal-agaf_new"]
    test_data = ["teacher_acc", "teacher_lbl_loss", "student_acc", "student_lbl_loss"]
    data_dict = {
        "train": {data: {depth: None for depth in depths} for data in train_data},
        "test": {data: {depth: None for depth in depths} for data in test_data}
    }
    save_obj(data_dict, f"./data/reliability_{dataset}/{model_name}_{data_size}_data.bin")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Evaluator(object):
    def __init__(self):
        self.GPU_ID = None
        self.block_num = None
        self.BATCH_SIZE = None
        self.PREVIOUS_LIST = None
        self.PATH = {}

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.train_loader = None
        self.test_loader = None
        raise NotImplementedError

    def _prepare_model(self):
        raise NotImplementedError

    def set_eval_mode(self):
        self.strong1.eval()
        self.strong2.eval()
        self.teacher.eval()
        self.strong1_outconv.eval()
        self.strong2_outconv.eval()
        self.teacher_outconv.eval()
        self.student.eval()
        self.common.eval()
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.g1.eval()
        self.g2.eval()
        self.g3.eval()
        if len(self.PREVIOUS_LIST) != 0:
            self.p_student.eval()
            self.p_common.eval()
            self.p_conv1.eval()
            self.p_conv2.eval()
            self.p_conv3.eval()
            self.p_g1.eval()
            self.p_g2.eval()
            self.p_g3.eval()

    def add_data(self, mode, data_type, value):
        DATA_DICT[mode][data_type][self.block_num] = value
        print(f" - {mode} {data_type}: {value}")

    def eval_train(self):
        self.set_eval_mode()

        meter_teacher = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_normal_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_normal_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_normal_remain = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_f_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_f_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_af_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_af_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_gaf_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_gaf_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_agaf_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_agaf_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_normal_agaf_current = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)
        meter_normal_agaf_new = AverageValueMeter(gpu_id=self.GPU_ID, dim=self.FEATURE_DIM)

        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc="Train set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                f_teacher = self.teacher(images)
                f_student = self.student(images)
                f = self.common(images)
                af = self.conv1(f)
                if len(self.PREVIOUS_LIST) != 0:
                    p_f_student = self.p_student(images)
                    p_f = self.p_common(images)
                    p_af = self.p_conv1(p_f)
                else:
                    p_f_student = 0
                    p_f = 0
                    p_af = 0

                gaf = gbf = gcf = None
                f_reliable = self.common(images)
                for _ in range(self.G_USE_LOOP):
                    gaf = self.g1(self.conv1(f_reliable))
                    gbf = self.g2(self.conv2(f_reliable))
                    gcf = self.g3(self.conv3(f_reliable))
                    f_reliable = (gaf + gbf + gcf) / 3
                agaf = self.conv1(f_reliable)

                if len(self.PREVIOUS_LIST) != 0:
                    p_gaf = p_gbf = p_gcf = None
                    p_f_reliable = self.p_common(images)
                    for _ in range(self.G_USE_LOOP):
                        p_gaf = self.p_g1(self.p_conv1(p_f_reliable))
                        p_gbf = self.p_g2(self.p_conv2(p_f_reliable))
                        p_gcf = self.p_g3(self.p_conv3(p_f_reliable))
                        p_f_reliable = (p_gaf + p_gbf + p_gcf) / 3
                    p_agaf = self.p_conv1(p_f_reliable)
                else:
                    p_gaf = 0
                    p_agaf = 0


                meter_teacher.add(f_teacher)
                meter_normal_current.add(f_student)
                meter_normal_new.add(f_student - p_f_student)
                meter_normal_remain.add(f_teacher - f_student)
                meter_f_current.add(f)
                meter_f_new.add(f - p_f)
                meter_af_current.add(af)
                meter_af_new.add(af - p_af)
                meter_gaf_current.add(gaf)
                meter_gaf_new.add(gaf - p_gaf)
                meter_agaf_current.add(agaf)
                meter_agaf_new.add(agaf - p_agaf)
                meter_normal_agaf_current.add(f_student - agaf)
                meter_normal_agaf_new.add((f_student - p_f_student) - (agaf - p_agaf))

            self.add_data("train", "var_teacher", meter_teacher.get_var().item())
            self.add_data("train", "var_normal_current", meter_normal_current.get_var().item())
            self.add_data("train", "var_normal_new", meter_normal_new.get_var().item())
            self.add_data("train", "var_normal_remain", meter_normal_remain.get_var().item())
            self.add_data("train", "var_f_current", meter_f_current.get_var().item())
            self.add_data("train", "var_f_new", meter_f_new.get_var().item())
            self.add_data("train", "var_af_current", meter_af_current.get_var().item())
            self.add_data("train", "var_af_new", meter_af_new.get_var().item())
            self.add_data("train", "var_gaf_current", meter_gaf_current.get_var().item())
            self.add_data("train", "var_gaf_new", meter_gaf_new.get_var().item())
            self.add_data("train", "var_agaf_current", meter_agaf_current.get_var().item())
            self.add_data("train", "var_agaf_new", meter_agaf_new.get_var().item())
            self.add_data("train", "var_normal-agaf_current", meter_normal_agaf_current.get_var().item())
            self.add_data("train", "var_normal-agaf_new", meter_normal_agaf_new.get_var().item())

    def eval_test(self):
        self.set_eval_mode()
        correct_student = 0
        correct_teacher = 0
        lbl_loss_student = 0
        lbl_loss_teacher = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Test set", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                pred_student = self.teacher_outconv(self.student(images))
                correct_student += pred_student.data.max(1)[1].eq(labels.data).sum()
                for i in range(labels.shape[0]):
                    lbl_loss_student += self.label_criterion(pred_student[i].unsqueeze(0), labels[i].unsqueeze(0)).data


                pred_teacher = self.teacher_outconv(self.teacher(images))
                correct_teacher += pred_teacher.data.max(1)[1].eq(labels.data).sum()
                for i in range(labels.shape[0]):
                    lbl_loss_teacher += self.label_criterion(pred_teacher[i].unsqueeze(0), labels[i].unsqueeze(0)).data

        acc_student = 100. * float(correct_student) / len(self.test_loader.dataset)
        acc_teacher = 100. * float(correct_teacher) / len(self.test_loader.dataset)
        lbl_loss_student = lbl_loss_student.item() / len(self.test_loader.dataset)
        lbl_loss_teacher = lbl_loss_teacher.item() / len(self.test_loader.dataset)
        self.add_data("test", "teacher_acc", acc_teacher)
        self.add_data("test", "teacher_lbl_loss", lbl_loss_teacher)
        self.add_data("test", "student_acc", acc_student)
        self.add_data("test", "student_lbl_loss", lbl_loss_student)

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
        self.DATA_ROOT = f"./data/CIFAR10"
        self.LOOP = args.loop
        self.G_USE_LOOP = args.use_loop
        self.NORMAL_MODEL_ROOT = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "disentangle")
        self.RELIABLE_MODEL_ROOT = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "reliability")
        self.G_MODEL_ROOT = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "reliability/g/loop{}".format(args.loop))
        self.PATH = {}

        self.PATH["strong_1"] = osp.join(PROJECT_ROOT, "pretrained_models/resnet44_cifar10/resnet44_strong_1_model.pth")
        self.PATH["strong_2"] = osp.join(PROJECT_ROOT, "pretrained_models/resnet44_cifar10/resnet44_strong_2_model.pth")
        self.PATH['teacher'] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_cifar10/{self.MODEL_SIZE}_model.pth")

        self.PATH['student'] = osp.join(self.NORMAL_MODEL_ROOT, "{}_model.pth".format(self.block_num))
        self.PATH["reliable_common"] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/current.pth".format(self.block_num))
        self.PATH['conv_1'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv1.pth".format(self.block_num))
        self.PATH['conv_2'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv2.pth".format(self.block_num))
        self.PATH['conv_3'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv3.pth".format(self.block_num))
        self.PATH['g_1'] = osp.join(self.G_MODEL_ROOT, "{}/ga.pth".format(self.block_num))
        self.PATH['g_2'] = osp.join(self.G_MODEL_ROOT, "{}/gb.pth".format(self.block_num))
        self.PATH['g_3'] = osp.join(self.G_MODEL_ROOT, "{}/gc.pth".format(self.block_num))

        if len(self.PREVIOUS_LIST) != 0:
            self.PATH['p_student'] = osp.join(self.NORMAL_MODEL_ROOT, "{}_model.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_reliable_common'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/current.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_conv_1'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv1.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_conv_2'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv2.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_conv_3'] = osp.join(self.RELIABLE_MODEL_ROOT, "{}/conv3.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_g_1'] = osp.join(self.G_MODEL_ROOT, "{}/ga.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_g_2'] = osp.join(self.G_MODEL_ROOT, "{}/gb.pth".format(self.PREVIOUS_LIST[-1]))
            self.PATH['p_g_3'] = osp.join(self.G_MODEL_ROOT, "{}/gc.pth".format(self.PREVIOUS_LIST[-1]))

    def _prepare_dataset(self):
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
        from net.resnet import resnet44
        self.strong1 = resnet44(disentangle=True).to(self.GPU_ID)
        self.strong1.load_state_dict(torch.load(self.PATH["strong_1"]))
        self.strong1_outconv = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            Flatten(),
            self.strong1.linear
        )
        self.strong2 = resnet44(disentangle=True).to(self.GPU_ID)
        self.strong2.load_state_dict(torch.load(self.PATH["strong_2"]))
        self.strong2_outconv = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            Flatten(),
            self.strong2.linear
        )
        if self.MODEL_NAME == "resnet8":
            from net.resnet import resnet8
            self.teacher = resnet8(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet14":
            from net.resnet import resnet14
            self.teacher = resnet14(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet20":
            from net.resnet import resnet20
            self.teacher = resnet20(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet32":
            from net.resnet import resnet32
            self.teacher = resnet32(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == "resnet44":
            from net.resnet import resnet44
            self.teacher = resnet44(disentangle=True).to(self.GPU_ID)
        self.teacher.load_state_dict(torch.load(self.PATH['teacher']))
        self.teacher_outconv = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
            Flatten(),
            self.teacher.linear
        )

        self.student = resnetdisentangler(self.block_num).to(self.GPU_ID)
        self.student.load_state_dict(torch.load(self.PATH['student']))
        self.common = resnetdisentangler(self.block_num).to(self.GPU_ID)
        self.common.load_state_dict(torch.load(self.PATH["reliable_common"]))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv1.load_state_dict(torch.load(self.PATH['conv_1']))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2.load_state_dict(torch.load(self.PATH['conv_2']))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3.load_state_dict(torch.load(self.PATH['conv_3']))
        self.g1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.g1.load_state_dict(torch.load(self.PATH['g_1']))
        self.g2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.g2.load_state_dict(torch.load(self.PATH['g_2']))
        self.g3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.g3.load_state_dict(torch.load(self.PATH['g_3']))

        if len(self.PREVIOUS_LIST) != 0:
            self.p_student = resnetdisentangler(self.PREVIOUS_LIST[-1]).to(self.GPU_ID)
            self.p_student.load_state_dict(torch.load(self.PATH['p_student']))
            self.p_common = resnetdisentangler(self.PREVIOUS_LIST[-1]).to(self.GPU_ID)
            self.p_common.load_state_dict(torch.load(self.PATH['p_reliable_common']))
            self.p_conv1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_conv1.load_state_dict(torch.load(self.PATH['p_conv_1']))
            self.p_conv2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_conv2.load_state_dict(torch.load(self.PATH['p_conv_2']))
            self.p_conv3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_conv3.load_state_dict(torch.load(self.PATH['p_conv_3']))
            self.p_g1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_g1.load_state_dict(torch.load(self.PATH['p_g_1']))
            self.p_g2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_g2.load_state_dict(torch.load(self.PATH['p_g_2']))
            self.p_g3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
            self.p_g3.load_state_dict(torch.load(self.PATH['p_g_3']))

        self.distill_criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)



if __name__ == '__main__':
    train_list = [1, 2, 4, 8, 16, 32]
    bs_list = [128, 128, 128, 128, 128, 128]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--block-num', type=int, default=None, help='train with n blocks')
    parser.add_argument('--previous-list', type=list, default=None, help='previous models')
    parser.add_argument('--loop', type=int, default=10)
    parser.add_argument('--use-loop', type=int, default=5)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--feature-dim', type=int, default=None)
    args = parser.parse_args()
    print(args)
    create_data_dict(args.dataset, args.model_name, args.data_size)
    DATA_DICT_PATH = f"./data/reliability_{args.dataset}/{args.model_name}_{args.data_size}_data.bin"
    DATA_DICT = load_obj(DATA_DICT_PATH)
    if args.dataset == "cifar10":
        args.feature_dim = 8 * 8 * 64
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = bs_list[i]
            print()
            print(args)
            evaluator = Eval_CIFAR10(args)
            evaluator.prepare()
            evaluator.run()
            save_obj(DATA_DICT, DATA_DICT_PATH)

    save_obj(DATA_DICT, DATA_DICT_PATH)