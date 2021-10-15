PROJECT_ROOT = "./"
import sys
sys.path.append(PROJECT_ROOT)

import argparse
import os
import os.path as osp
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


from tools import save_obj, load_obj, update_lr
from net.resnet import resnetdisentangler
from net.resnetbig import resnetdisentanglerbig

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

def get_current_big(cur_block_num, previous_path):
    current = resnetdisentanglerbig(cur_block_num)
    current_dict = current.state_dict()
    previous_dict = torch.load(previous_path)
    load_dict = {k: v for k, v in previous_dict.items() if k in current_dict}
    # for k, v in load_dict.items():
    #     print(k)
    current_dict.update(load_dict)
    current.load_state_dict(current_dict)
    return current


class ReconstructionTrainer(object):
    def __init__(self):
        self.GPU_ID = None
        self.block_num = None
        self.BATCH_SIZE = None
        self.LEARNING_RATE = None
        self.NUM_EPOCHS = None
        self.PREVIOUS_LIST = None
        self.LR_LIST = None
        self.EPOCH = 1
        self.PATH = {}

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()
        self._generate_plot_dic()

    def _generate_plot_dic(self):
        self.plot_dic = {
            "train_acc_1": [],
            "train_acc_2": [],
            "train_acc_3": [],
            "test_acc_1": [],
            "test_acc_2": [],
            "test_acc_3": [],
            "distill_loss_1": [],
            "distill_loss_2": [],
            "distill_loss_3": [],
            "label_loss_train_pretrained_1": [],
            "label_loss_train_pretrained_2": [],
            "label_loss_train_pretrained_3": [],
            "label_loss_test_pretrained_1": [],
            "label_loss_test_pretrained_2": [],
            "label_loss_test_pretrained_3": [],
            "label_loss_train_1": [],
            "label_loss_test_1": [],
            "label_loss_train_2": [],
            "label_loss_test_2": [],
            "label_loss_train_3": [],
            "label_loss_test_3": []
        }

    def _prepare_dataset(self):
        self.train_loader = None
        self.test_loader = None
        raise NotImplementedError

    def _prepare_pretrained(self):
        self.pretrained_model = None
        self.out_conv = None
        raise NotImplementedError

    def _prepare_model(self):
        self.optimizer = None
        self.criterion = None
        self.label_criterion = None
        self.current = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        raise NotImplementedError

    def get_pretrained(self, x, i):
        return self.pretrained_model[i](x).detach()

    def train_epoch(self):
        self.current.train()
        self.conv1.train()
        self.conv2.train()
        self.conv3.train()
        for i in range(3):
            self.out_conv[i].eval()
            self.pretrained_model[i].eval()
        step = 0
        distill_loss = torch.zeros(3)
        label_loss_pretrained = torch.zeros(3)
        label_loss_preconv = torch.zeros(3)
        correct_preconv = torch.zeros(3)
        correct_pretrained = torch.zeros(3)
        acc_preconv= torch.zeros(3)
        acc_pretrained= torch.zeros(3)

        print("Learning rate is", self.LR_LIST[self.EPOCH - 1])
        update_lr(self.optimizer, [self.LR_LIST[self.EPOCH - 1]])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
            self.optimizer.zero_grad()
            pretrained_1 = self.get_pretrained(images,0).detach()
            pretrained_2 = self.get_pretrained(images,1).detach()
            pretreined_3 = self.get_pretrained(images,2).detach()
            pretrained = torch.stack((pretrained_1, pretrained_2, pretreined_3),4)

            current_f = self.current(images)
            output_1 = self.conv1(current_f)
            output_2 = self.conv2(current_f)
            output_3 = self.conv3(current_f)
            output = torch.stack((output_1, output_2, output_3),4)

            loss=torch.zeros(3)
            total_loss = 0
            for i in range(3):
                loss[i] = self.criterion(output[:,:,:,:,i], pretrained[:,:,:,:,i])
                total_loss += loss[i]
                distill_loss[i] += loss[i].data

            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            for i in range(3):
                labels_pretrained = self.out_conv[i](F.relu(pretrained[:,:,:,:,i])).detach()
                label_loss_pretrained[i] += self.label_criterion(labels_pretrained, labels).data
                correct_pretrained[i] += labels_pretrained.data.max(1)[1].eq(labels.data).sum()

            for i in range(3):
                out = self.out_conv[i](output[:, :, :,:, i])
                y_pred = out.data.max(1)[1]
                correct_preconv[i] += y_pred.eq(labels.data).sum()
                labels_predict = out.detach()
                label_loss_preconv[i] += self.label_criterion(labels_predict, labels).data

            step += 1
            if step % 100 == 0:
                print(' distillation loss: {:.4f}'.format(total_loss.item()))

        length = len(self.train_loader.sampler) // self.BATCH_SIZE

        for i in range(3):
            acc_pretrained[i] = correct_pretrained[i] / len(self.train_loader.sampler)
            label_loss_pretrained[i] = label_loss_pretrained[i] / length
            acc_preconv[i] = 100. * float(correct_preconv[i]) / len(self.train_loader.sampler)
            distill_loss[i] = distill_loss[i] / length
            label_loss_preconv[i] = label_loss_preconv[i] / length


        self.plot_dic['label_loss_train_pretrained_1'].append(label_loss_pretrained[0])
        self.plot_dic['label_loss_train_pretrained_2'].append(label_loss_pretrained[1])
        self.plot_dic['label_loss_train_pretrained_3'].append(label_loss_pretrained[2])
        self.plot_dic['distill_loss_1'].append(distill_loss[0])
        self.plot_dic['distill_loss_2'].append(distill_loss[1])
        self.plot_dic['distill_loss_3'].append(distill_loss[2])
        self.plot_dic['train_acc_1'].append(acc_preconv[0])
        self.plot_dic['train_acc_2'].append(acc_preconv[1])
        self.plot_dic['train_acc_3'].append(acc_preconv[2])
        self.plot_dic['label_loss_train_1'].append(label_loss_preconv[0])
        self.plot_dic['label_loss_train_2'].append(label_loss_preconv[1])
        self.plot_dic['label_loss_train_3'].append(label_loss_preconv[2])
        print("Train Set: distillation Loss (feature): {0:2.3f}; Acc (add out linear): {1:.3f}%"
              .format(distill_loss.sum()/3, acc_preconv.sum()/3))

    def eval_epoch(self):
        self.current.eval()
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        for i in range(3):
            self.out_conv[i].eval()
            self.pretrained_model[i].eval()

        distill_correct = torch.zeros(3)
        pretrained_correct = torch.zeros(3)
        pretrained_acc = torch.zeros(3)
        label_loss_pretrained = torch.zeros(3)
        label_loss = torch.zeros(3)
        distill_acc = torch.zeros(3)
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                output_1 = self.conv1(self.current(images))
                output_2 = self.conv2(self.current(images))
                output_3 = self.conv3(self.current(images))
                output = torch.stack((output_1, output_2, output_3), 4)

                for i in range(3):
                    pretrained_prediction = self.out_conv[i](F.relu(self.get_pretrained(images,i)))
                    pretrained_correct[i] += pretrained_prediction.data.max(1)[1].eq(labels.data).sum()
                    label_loss_pretrained[i] += self.label_criterion(pretrained_prediction, labels)

                    prediction = self.out_conv[i](output[:,:,:,:,i])
                    distill_correct[i] += prediction.data.max(1)[1].eq(labels.data).sum()
                    label_loss[i] += self.label_criterion(prediction, labels)

        length = len(self.test_loader.dataset) // self.BATCH_SIZE

        for i in range(3):
            pretrained_acc[i] = 100. * float(pretrained_correct[i]) / len(self.test_loader.dataset)
            label_loss_pretrained[i] = label_loss_pretrained[i] / length

            distill_acc[i] = 100. * float(distill_correct[i]) / len(self.test_loader.dataset)
            label_loss[i] = label_loss[i] / length

        self.plot_dic['label_loss_test_pretrained_1'].append(label_loss_pretrained[0])
        self.plot_dic['label_loss_test_pretrained_2'].append(label_loss_pretrained[1])
        self.plot_dic['label_loss_test_pretrained_3'].append(label_loss_pretrained[2])
        self.plot_dic['test_acc_1'].append(distill_acc[0])
        self.plot_dic['test_acc_2'].append(distill_acc[1])
        self.plot_dic['test_acc_3'].append(distill_acc[2])
        self.plot_dic['label_loss_test_1'].append(label_loss[0])
        self.plot_dic['label_loss_test_2'].append(label_loss[1])
        self.plot_dic['label_loss_test_3'].append(label_loss[2])

        print('Accuracy of the network on the test images: {} %'.format(distill_acc.sum()/3))
        print('Accuracy of the network on the test images (pretrained model): {} %'.format(pretrained_acc.sum()/3))
        return distill_acc, pretrained_acc

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            self.train_epoch()
            self.draw()
            if self.EPOCH % 100 == 0:
                self.save()

        torch.save(self.current.to("cpu").state_dict(), self.PATH['model_save_path'])
        torch.save(self.conv1.to("cpu").state_dict(), self.PATH['model_save_path_conv1'])
        torch.save(self.conv2.to("cpu").state_dict(), self.PATH['model_save_path_conv2'])
        torch.save(self.conv3.to("cpu").state_dict(), self.PATH['model_save_path_conv3'])

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(16, 12))
        # train & test accuracy
        plt.subplot(2, 4, 1)
        x = range(1, len(self.plot_dic["train_acc_1"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["train_acc_1"], label="train_acc_1")
        # plt.plot(x, self.plot_dic["test_acc_1"], label="test_acc_1")
        plt.legend()
        plt.subplot(2, 4, 2)
        x = range(1, len(self.plot_dic["train_acc_2"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["train_acc_2"], label="train_acc_2")
        # plt.plot(x, self.plot_dic["test_acc_2"], label="test_acc_2")
        plt.legend()
        plt.subplot(2, 4, 3)
        x = range(1, len(self.plot_dic["train_acc_3"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["train_acc_3"], label="train_acc_3")
        # plt.plot(x, self.plot_dic["test_acc_3"], label="test_acc_3")
        plt.legend()

        # distillation loss (MSE) on training set
        plt.subplot(2, 4, 4)
        x = range(1, len(self.plot_dic["distill_loss_1"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["distill_loss_1"], label="distill_loss_1")
        plt.plot(x, self.plot_dic["distill_loss_2"], label="distill_loss_2")
        plt.plot(x, self.plot_dic["distill_loss_3"], label="distill_loss_3")
        plt.legend()
        # label loss (CrossEntropy) on training/testing set
        plt.subplot(2, 4, 5)
        x = range(1, len(self.plot_dic["label_loss_train_pretrained_1"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["label_loss_train_pretrained_1"], label="label_loss_train_pretrained_1")
        plt.plot(x, self.plot_dic["label_loss_train_1"], label="label_loss_train_1")
        # plt.plot(x, self.plot_dic["label_loss_test_pretrained_1"], label="label_loss_test_pretrained_1")
        # plt.plot(x, self.plot_dic["label_loss_test_1"], label="label_loss_test_1")
        plt.legend()

        plt.subplot(2, 4, 6)
        x = range(1, len(self.plot_dic["label_loss_train_pretrained_2"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["label_loss_train_pretrained_2"], label="label_loss_train_pretrained_2")
        plt.plot(x, self.plot_dic["label_loss_train_2"], label="label_loss_train_2")
        # plt.plot(x, self.plot_dic["label_loss_test_pretrained_2"], label="label_loss_test_pretrained_2")
        # plt.plot(x, self.plot_dic["label_loss_test_2"], label="label_loss_test_2")
        plt.legend()

        plt.subplot(2, 4, 7)
        x = range(1, len(self.plot_dic["label_loss_train_pretrained_3"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["label_loss_train_pretrained_3"], label="label_loss_train_pretrained_3")
        plt.plot(x, self.plot_dic["label_loss_train_3"], label="label_loss_train_3")
        # plt.plot(x, self.plot_dic["label_loss_test_pretrained_3"], label="label_loss_test_pretrained_3")
        # plt.plot(x, self.plot_dic["label_loss_test_3"], label="label_loss_test_3")
        plt.legend()

        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        torch.save(self.current.state_dict(), self.PATH['model_save_path'])
        torch.save(self.conv1.state_dict(), self.PATH['model_save_path_conv1'])
        torch.save(self.conv2.state_dict(), self.PATH['model_save_path_conv2'])
        torch.save(self.conv3.state_dict(), self.PATH['model_save_path_conv3'])

        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


class Rec_CIFAR10(ReconstructionTrainer):
    def __init__(self, args):
        super(Rec_CIFAR10, self).__init__()
        self.block_num = args.block_num
        self.DATA_ROOT = "./data/CIFAR10"
        self.BATCH_SIZE = args.batch_size
        self.LR_LIST = np.logspace(args.rec_initial_lr, args.rec_final_lr, args.rec_num_epochs)
        self.NUM_EPOCHS = args.rec_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{self.MODEL_NAME}_{self.DATA_SIZE}"
        self.GPU_ID = args.gpu_id
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "reliability")
        self.PATH['model_save_folder'] = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.block_num))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH['model_save_path'] = osp.join(self.PATH['model_save_folder'], "current.pth")
        self.PATH['model_save_path_conv1'] = osp.join(self.PATH['model_save_folder'], "conv1.pth")
        self.PATH['model_save_path_conv2'] = osp.join(self.PATH['model_save_folder'], "conv2.pth")
        self.PATH['model_save_path_conv3'] = osp.join(self.PATH['model_save_folder'], "conv3.pth")

        self.PATH['plot_data_save_path'] = osp.join(self.PATH['model_save_folder'], "data.bin")
        self.PATH['strong_1_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet44_cifar10/resnet44_strong_1_model.pth")
        self.PATH['strong_2_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet44_cifar10/resnet44_strong_2_model.pth")
        self.PATH['teacher_path'] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_cifar10/{self.MODEL_SIZE}_model.pth")

        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_path"] = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.PREVIOUS_LIST[-1]), "current.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH['model_save_folder']):
            os.makedirs(self.PATH['model_save_folder'])

    def _prepare_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
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
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.DATA_ROOT, train=False, transform=transform_test),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    def _prepare_pretrained(self):
        from net.resnet import resnet44
        pretrained = []
        out_conv = []

        for i in range(3):
            if i == 0:
                if self.MODEL_NAME == "resnet8":
                    from net.resnet import resnet8
                    pretrained_i = resnet8(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet14":
                    from net.resnet import resnet14
                    pretrained_i = resnet14(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet20":
                    from net.resnet import resnet20
                    pretrained_i = resnet20(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet32":
                    from net.resnet import resnet32
                    pretrained_i = resnet32(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet44":
                    from net.resnet import resnet44
                    pretrained_i = resnet44(disentangle=True).to(self.GPU_ID)
                pretrained_i.load_state_dict(torch.load(self.PATH['teacher_path']))
            else:
                pretrained_i = resnet44(disentangle=True).to(self.GPU_ID)
                pretrained_i.load_state_dict(torch.load(self.PATH['strong_{}_path'.format(i)]))
            pretrained.append(pretrained_i.to(self.GPU_ID))

            out_conv.append(
                nn.Sequential(
                    nn.AvgPool2d(8, 8),
                    Flatten(),
                    pretrained_i.linear
                ).to(self.GPU_ID)
            )

        return pretrained, out_conv

    def _prepare_model(self):
        from net.resnet import resnetdisentangler
        self.pretrained_model, self.out_conv = self._prepare_pretrained()
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentangler(self.block_num).to(self.GPU_ID)
        else:
            self.current = get_current(self.block_num, self.PATH["previous_path"]).to(self.GPU_ID)
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(
            [{'params': self.current.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv1.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv2.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv3.parameters(), 'lr': self.LEARNING_RATE}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


class Rec_CUB(ReconstructionTrainer):
    def __init__(self, args):
        super(Rec_CUB, self).__init__()
        self.block_num = args.block_num
        self.DATA_ROOT = "./data/CUB/CUB_strong"
        self.BATCH_SIZE = args.batch_size
        self.LR_LIST = np.logspace(args.rec_initial_lr, args.rec_final_lr, args.rec_num_epochs)
        self.NUM_EPOCHS = args.rec_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{self.MODEL_NAME}_{self.DATA_SIZE}"
        self.GPU_ID = args.gpu_id
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_cub", self.MODEL_NAME, self.MODEL_SIZE, "reliability")
        self.PATH['model_save_folder'] = osp.join(PROJECT_ROOT, "models/models_cub", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.block_num))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH['model_save_path'] = osp.join(self.PATH['model_save_folder'], "current.pth")
        self.PATH['model_save_path_conv1'] = osp.join(self.PATH['model_save_folder'], "conv1.pth")
        self.PATH['model_save_path_conv2'] = osp.join(self.PATH['model_save_folder'], "conv2.pth")
        self.PATH['model_save_path_conv3'] = osp.join(self.PATH['model_save_folder'], "conv3.pth")

        self.PATH['plot_data_save_path'] = osp.join(self.PATH['model_save_folder'], "data.bin")
        self.PATH['strong_1_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet34_cub/resnet34_strong_1_model.pth")
        self.PATH['strong_2_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet34_cub/resnet34_strong_2_model.pth")
        self.PATH['teacher_path'] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_cub/{self.MODEL_SIZE}_model.pth")

        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_path"] = osp.join(PROJECT_ROOT, "models/models_cub", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.PREVIOUS_LIST[-1]), "current.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH['model_save_folder']):
            os.makedirs(self.PATH['model_save_folder'])

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
        sampler_train = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, sampler=sampler_train)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_pretrained(self):
        from net.resnetbig import resnet34_cub
        pretrained = []
        out_conv = []

        for i in range(3):
            if i == 0:
                if self.MODEL_NAME == "resnet18":
                    from net.resnetbig import resnet18_cub
                    pretrained_i = resnet18_cub(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet34":
                    pretrained_i = resnet34_cub(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "vgg16":
                    from net.vggbig import vgg16_cub
                    pretrained_i = vgg16_cub(disentangle=True).to(self.GPU_ID)
                else:
                    raise Exception("Failed in loading model.")
                pretrained_i.load_state_dict(torch.load(self.PATH['teacher_path']))
                print(f"{self.PATH['teacher_path']} loaded.")
            else:
                pretrained_i = resnet34_cub(disentangle=True).to(self.GPU_ID)
                pretrained_i.load_state_dict(torch.load(self.PATH['strong_{}_path'.format(i)]))
            pretrained.append(pretrained_i.to(self.GPU_ID))

            if i == 0:
                if self.MODEL_NAME == "vgg16":
                    out_conv.append(
                        nn.Sequential(
                            pretrained_i.avgpool,
                            Flatten(),
                            pretrained_i.classifier
                        ).to(self.GPU_ID)
                    )
                else:
                    out_conv.append(
                        nn.Sequential(
                            pretrained_i.avgpool,
                            Flatten(),
                            pretrained_i.fc
                        ).to(self.GPU_ID)
                    )
            else:
                out_conv.append(
                    nn.Sequential(
                        pretrained_i.avgpool,
                        Flatten(),
                        pretrained_i.fc
                    ).to(self.GPU_ID)
                )

        return pretrained, out_conv

    def _prepare_model(self):
        self.pretrained_model, self.out_conv = self._prepare_pretrained()
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentanglerbig(self.block_num).to(self.GPU_ID)
        else:
            self.current = get_current_big(self.block_num, self.PATH["previous_path"]).to(self.GPU_ID)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(
            [{'params': self.current.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv1.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv2.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv3.parameters(), 'lr': self.LEARNING_RATE}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


class Rec_DOGS(ReconstructionTrainer):
    def __init__(self, args):
        super(Rec_DOGS, self).__init__()
        self.block_num = args.block_num
        self.DATA_ROOT = "./data/DOGS/DOGS_strong"
        self.BATCH_SIZE = args.batch_size
        self.LR_LIST = np.logspace(args.rec_initial_lr, args.rec_final_lr, args.rec_num_epochs)
        self.NUM_EPOCHS = args.rec_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{self.MODEL_NAME}_{self.DATA_SIZE}"
        self.GPU_ID = args.gpu_id
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_dogs", self.MODEL_NAME, self.MODEL_SIZE, "reliability")
        self.PATH['model_save_folder'] = osp.join(PROJECT_ROOT, "models/models_dogs", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.block_num))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH['model_save_path'] = osp.join(self.PATH['model_save_folder'], "current.pth")
        self.PATH['model_save_path_conv1'] = osp.join(self.PATH['model_save_folder'], "conv1.pth")
        self.PATH['model_save_path_conv2'] = osp.join(self.PATH['model_save_folder'], "conv2.pth")
        self.PATH['model_save_path_conv3'] = osp.join(self.PATH['model_save_folder'], "conv3.pth")

        self.PATH['plot_data_save_path'] = osp.join(self.PATH['model_save_folder'], "data.bin")
        self.PATH['strong_1_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet34_dogs/resnet34_strong_1_model.pth")
        self.PATH['strong_2_path'] = osp.join(PROJECT_ROOT, "pretrained_models/resnet34_dogs/resnet34_strong_2_model.pth")
        self.PATH['teacher_path'] = osp.join(PROJECT_ROOT, f"pretrained_models/{self.MODEL_NAME}_dogs/{self.MODEL_SIZE}_model.pth")

        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_path"] = osp.join(PROJECT_ROOT, "models/models_dogs", self.MODEL_NAME, self.MODEL_SIZE, "reliability", str(self.PREVIOUS_LIST[-1]), "current.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH['model_save_folder']):
            os.makedirs(self.PATH['model_save_folder'])

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
        sampler_train = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, sampler=sampler_train)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DOGS_mean, DOGS_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_pretrained(self):
        from net.resnetbig import resnet34_dogs
        pretrained = []
        out_conv = []

        for i in range(3):
            if i == 0:
                if self.MODEL_NAME == "resnet18":
                    from net.resnetbig import resnet18_dogs
                    pretrained_i = resnet18_dogs(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "resnet34":
                    pretrained_i = resnet34_dogs(disentangle=True).to(self.GPU_ID)
                elif self.MODEL_NAME == "vgg16":
                    from net.vggbig import vgg16_dogs
                    pretrained_i = vgg16_dogs(disentangle=True).to(self.GPU_ID)
                else:
                    raise Exception("Failed in loading model.")
                pretrained_i.load_state_dict(torch.load(self.PATH['teacher_path']))
                print(f"{self.PATH['teacher_path']} loaded.")
            else:
                pretrained_i = resnet34_dogs(disentangle=True).to(self.GPU_ID)
                pretrained_i.load_state_dict(torch.load(self.PATH['strong_{}_path'.format(i)]))
            pretrained.append(pretrained_i.to(self.GPU_ID))

            if i == 0:
                if self.MODEL_NAME == "vgg16":
                    out_conv.append(
                        nn.Sequential(
                            pretrained_i.avgpool,
                            Flatten(),
                            pretrained_i.classifier
                        ).to(self.GPU_ID)
                    )
                else:
                    out_conv.append(
                        nn.Sequential(
                            pretrained_i.avgpool,
                            Flatten(),
                            pretrained_i.fc
                        ).to(self.GPU_ID)
                    )
            else:
                out_conv.append(
                    nn.Sequential(
                        pretrained_i.avgpool,
                        Flatten(),
                        pretrained_i.fc
                    ).to(self.GPU_ID)
                )

        return pretrained, out_conv

    def _prepare_model(self):
        self.pretrained_model, self.out_conv = self._prepare_pretrained()
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentanglerbig(self.block_num).to(self.GPU_ID)
        else:
            self.current = get_current_big(self.block_num, self.PATH["previous_path"]).to(self.GPU_ID)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(
            [{'params': self.current.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv1.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv2.parameters(), 'lr': self.LEARNING_RATE},
             {'params': self.conv3.parameters(), 'lr': self.LEARNING_RATE}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)



class GTrainer(object):
    def __init__(self):
        self.GPU_ID = None
        self.block_num = None
        self.BATCH_SIZE = None
        self.G_LR_LIST = None
        self.NUM_EPOCHS = None
        self.PREVIOUS_LIST = None
        self.LOOP = None
        self.EPOCH = 1
        self.PATH = {}

    def prepare(self):
        self._prepare_g()
        self._prepare_model()
        self._prepare_dataset()
        self._generate_plot_dic()

    def _generate_plot_dic(self):
        self.plot_dic = {}
        for i in range(1, self.LOOP + 1):
            self.plot_dic["loss_loop_{}".format(i)]=[]
            self.plot_dic["A_loss_loop_{}".format(i)] = []
            self.plot_dic["B_loss_loop_{}".format(i)] = []
            self.plot_dic["C_loss_loop_{}".format(i)] = []

    def _prepare_dataset(self):
        self.train_loader = None
        self.test_loader = None
        raise NotImplementedError

    def _prepare_g(self):
        self.ga = None
        self.gb = None
        self.gc = None
        raise NotImplementedError

    def _prepare_model(self):
        self.current = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.g_optimizer = None
        self.criterion = None
        raise NotImplementedError

    def train_g(self, loop):
        self.current.eval()
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.ga.train()
        self.gb.train()
        self.gc.train()

        step = 0
        total_loss = 0
        a_loss = 0
        b_loss = 0
        c_loss = 0

        update_lr(self.g_optimizer, [self.G_LR_LIST[self.EPOCH - 1]])

        print("G learning rate is", self.g_optimizer.param_groups[0]["lr"])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
            self.g_optimizer.zero_grad()

            f = self.current(images).detach()

            for _ in range(1, loop): # loop=1: no effect, loop=2: once, loop=3: twice
                a = self.ga(self.conv1(f)).detach()
                b = self.gb(self.conv2(f)).detach()
                c = self.gc(self.conv3(f)).detach()
                f = (a + b + c) / 3
                f = f.detach()

            a_hat = self.ga(self.conv1(f).detach())
            b_hat = self.gb(self.conv2(f).detach())
            c_hat = self.gc(self.conv3(f).detach())

            loss_a = self.criterion(a_hat, f)
            loss_b = self.criterion(b_hat, f)
            loss_c = self.criterion(c_hat, f)

            loss = loss_a + loss_b + loss_c

            loss.backward()
            # print(self.conv1[0].weight.grad) # check
            self.g_optimizer.step()

            a_loss += loss_a.data
            b_loss += loss_b.data
            c_loss += loss_c.data
            total_loss += loss.data

            step += 1

        length = len(self.train_loader.sampler) // self.BATCH_SIZE # revised
        a_loss = a_loss / length
        b_loss = b_loss / length
        c_loss = c_loss / length
        total_loss = total_loss / length

        self.plot_dic["A_loss_loop_{}".format(loop)].append(a_loss)
        self.plot_dic["B_loss_loop_{}".format(loop)].append(b_loss)
        self.plot_dic["C_loss_loop_{}".format(loop)].append(c_loss)
        self.plot_dic["loss_loop_{}".format(loop)].append(total_loss)

        print("[loss A]={0:.3f}, [loss B]={1:.3f}, [loss C]={2:.3f}, [loss]={3:.3f}".format(a_loss, b_loss, c_loss, total_loss))

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            for loop in range(1, self.LOOP+1):
                self.train_g(loop)
                if self.EPOCH % 50 == 0:
                    self.save()
            self.draw()
        torch.save(self.ga.to("cpu").state_dict(), self.PATH["ga_save_path"])
        torch.save(self.gb.to("cpu").state_dict(), self.PATH["gb_save_path"])
        torch.save(self.gc.to("cpu").state_dict(), self.PATH["gc_save_path"])

        # torch.save(self.out_conv_student.to("cpu").state_dict(), self.PATH['outconv_save_path'])

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(16, 8))
        for i in range(1, self.LOOP+1):
            plt.subplot(4, self.LOOP/2, i)
            x = range(1, len(self.plot_dic["loss_loop_{}".format(i)]) + 1)
            plt.xlabel("epoch")
            plt.plot(x, self.plot_dic["loss_loop_{}".format(i)], label="loss_loop_{}".format(i))
            plt.legend()
        for i in range(1, self.LOOP+1):
            plt.subplot(4, self.LOOP/2, i+self.LOOP)
            x = range(1, len(self.plot_dic["A_loss_loop_{}".format(i)]) + 1)
            plt.xlabel("epoch")
            plt.plot(x, self.plot_dic["A_loss_loop_{}".format(i)], label="A_loss_loop_{}".format(i))
            plt.plot(x, self.plot_dic["B_loss_loop_{}".format(i)], label="B_loss_loop_{}".format(i))
            plt.plot(x, self.plot_dic["C_loss_loop_{}".format(i)], label="C_loss_loop_{}".format(i))
            plt.legend()


        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        torch.save(self.ga.state_dict(), self.PATH["ga_save_path"])
        torch.save(self.gb.state_dict(), self.PATH["gb_save_path"])
        torch.save(self.gc.state_dict(), self.PATH["gc_save_path"])

        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


class G_CIFAR10(GTrainer):
    def __init__(self, args):
        super(G_CIFAR10, self).__init__()
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.G_LR_LIST = np.logspace(args.g_initial_lr, args.g_final_lr, args.g_num_epochs)
        self.NUM_EPOCHS = args.g_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{args.model_name}_{args.data_size}"
        self.LOOP = args.loop
        self.DATA_ROOT = "./data/CIFAR10"
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_cifar10", self.MODEL_NAME, self.MODEL_SIZE,
                                                "reliability/g/loop{}".format(self.LOOP))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH["g_save_folder"] = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability/g/loop{}".format(self.LOOP), str(self.block_num))
        self.PATH["ga_save_path"] = osp.join(self.PATH["g_save_folder"], "ga.pth")
        self.PATH["gb_save_path"] = osp.join(self.PATH["g_save_folder"], "gb.pth")
        self.PATH["gc_save_path"] = osp.join(self.PATH["g_save_folder"], "gc.pth")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["g_save_folder"], "data.bin.pth")
        self.PATH["student_folder"] = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability", str(self.block_num))
        self.PATH["current_path"] = osp.join(self.PATH["student_folder"], "current.pth")
        self.PATH["conv1_path"] = osp.join(self.PATH["student_folder"], "conv1.pth")
        self.PATH["conv2_path"] = osp.join(self.PATH["student_folder"], "conv2.pth")
        self.PATH["conv3_path"] = osp.join(self.PATH["student_folder"], "conv3.pth")
        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_g_folder"] = osp.join(PROJECT_ROOT, "models/models_cifar10", self.MODEL_NAME, self.MODEL_SIZE,
                                                      "reliability/g/loop{}".format(self.LOOP), str(self.PREVIOUS_LIST[-1]))
            self.PATH["previous_ga_path"] = osp.join(self.PATH["previous_g_folder"], "ga.pth")
            self.PATH["previous_gb_path"] = osp.join(self.PATH["previous_g_folder"], "gb.pth")
            self.PATH["previous_gc_path"] = osp.join(self.PATH["previous_g_folder"], "gc.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH["g_save_folder"]):
            os.makedirs(self.PATH["g_save_folder"])

    def _prepare_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_train = datasets.CIFAR10(self.DATA_ROOT, train=True, transform=transform_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.BATCH_SIZE,
            sampler=sampler_train,
            num_workers=1
        )
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.DATA_ROOT, train=False, transform=transform_test),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    def _prepare_model(self):
        self.current = resnetdisentangler(self.block_num)
        self.current.load_state_dict(torch.load(self.PATH["current_path"]))
        self.current.to(self.GPU_ID)
        print("{}/current.pth loaded".format(self.block_num))
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv1.load_state_dict(torch.load(self.PATH["conv1_path"]))
        self.conv2.load_state_dict(torch.load(self.PATH["conv2_path"]))
        self.conv3.load_state_dict(torch.load(self.PATH["conv3_path"]))

        self.g_optimizer = torch.optim.Adam(
            [{'params': self.ga.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gb.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gc.parameters(), 'lr': self.G_LR_LIST[0]}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.criterion = nn.MSELoss()

    def _prepare_g(self):
        self.ga = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False))
        self.gb = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False))
        self.gc = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False))
        if len(self.PREVIOUS_LIST) != 0:
            print(self.PATH['previous_g_folder'] + " loaded.")
            self.ga.load_state_dict(torch.load(self.PATH["previous_ga_path"]))
            self.gb.load_state_dict(torch.load(self.PATH["previous_gb_path"]))
            self.gc.load_state_dict(torch.load(self.PATH["previous_gc_path"]))
        self.ga.to(self.GPU_ID)
        self.gb.to(self.GPU_ID)
        self.gc.to(self.GPU_ID)


class G_CUB(GTrainer):
    def __init__(self, args):
        super(G_CUB, self).__init__()
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.G_LR_LIST = np.logspace(args.g_initial_lr, args.g_final_lr, args.g_num_epochs)
        self.NUM_EPOCHS = args.g_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{args.model_name}_{args.data_size}"
        self.LOOP = args.loop
        self.DATA_ROOT = "./data/CUB/CUB_strong"
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_cub", self.MODEL_NAME, self.MODEL_SIZE,
                                                "reliability/g/loop{}".format(self.LOOP))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH["g_save_folder"] = osp.join(PROJECT_ROOT, "models/models_cub", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability/g/loop{}".format(self.LOOP), str(self.block_num))
        self.PATH["ga_save_path"] = osp.join(self.PATH["g_save_folder"], "ga.pth")
        self.PATH["gb_save_path"] = osp.join(self.PATH["g_save_folder"], "gb.pth")
        self.PATH["gc_save_path"] = osp.join(self.PATH["g_save_folder"], "gc.pth")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["g_save_folder"], "data.bin.pth")
        self.PATH["student_folder"] = osp.join(PROJECT_ROOT, "models/models_cub", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability", str(self.block_num))
        self.PATH["current_path"] = osp.join(self.PATH["student_folder"], "current.pth")
        self.PATH["conv1_path"] = osp.join(self.PATH["student_folder"], "conv1.pth")
        self.PATH["conv2_path"] = osp.join(self.PATH["student_folder"], "conv2.pth")
        self.PATH["conv3_path"] = osp.join(self.PATH["student_folder"], "conv3.pth")
        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_g_folder"] = osp.join(PROJECT_ROOT, "models/models_cub", self.MODEL_NAME, self.MODEL_SIZE,
                                                      "reliability/g/loop{}".format(self.LOOP), str(self.PREVIOUS_LIST[-1]))
            self.PATH["previous_ga_path"] = osp.join(self.PATH["previous_g_folder"], "ga.pth")
            self.PATH["previous_gb_path"] = osp.join(self.PATH["previous_g_folder"], "gb.pth")
            self.PATH["previous_gc_path"] = osp.join(self.PATH["previous_g_folder"], "gc.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH["g_save_folder"]):
            os.makedirs(self.PATH["g_save_folder"])

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
        sampler_train = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, sampler=sampler_train)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CUB_mean, CUB_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_model(self):
        self.current = resnetdisentanglerbig(self.block_num)
        self.current.load_state_dict(torch.load(self.PATH["current_path"]))
        self.current.to(self.GPU_ID)
        print(f' - current {self.PATH["current_path"]}')
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv1.load_state_dict(torch.load(self.PATH["conv1_path"]))
        self.conv2.load_state_dict(torch.load(self.PATH["conv2_path"]))
        self.conv3.load_state_dict(torch.load(self.PATH["conv3_path"]))
        print(f' - conv1   {self.PATH["conv1_path"]}')
        print(f' - conv2   {self.PATH["conv2_path"]}')
        print(f' - conv3   {self.PATH["conv3_path"]}')

        self.g_optimizer = torch.optim.Adam(
            [{'params': self.ga.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gb.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gc.parameters(), 'lr': self.G_LR_LIST[0]}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()

    def _prepare_g(self):
        self.ga = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        self.gb = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        self.gc = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        if len(self.PREVIOUS_LIST) != 0:
            print(self.PATH['previous_g_folder'] + " loaded.")
            self.ga.load_state_dict(torch.load(self.PATH["previous_ga_path"]))
            self.gb.load_state_dict(torch.load(self.PATH["previous_gb_path"]))
            self.gc.load_state_dict(torch.load(self.PATH["previous_gc_path"]))
        self.ga.to(self.GPU_ID)
        self.gb.to(self.GPU_ID)
        self.gc.to(self.GPU_ID)


class G_DOGS(GTrainer):
    def __init__(self, args):
        super(G_DOGS, self).__init__()
        self.GPU_ID = args.gpu_id
        self.block_num = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.G_LR_LIST = np.logspace(args.g_initial_lr, args.g_final_lr, args.g_num_epochs)
        self.NUM_EPOCHS = args.g_num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.MODEL_SIZE = f"{args.model_name}_{args.data_size}"
        self.LOOP = args.loop
        self.DATA_ROOT = "./data/DOGS/DOGS_strong"
        self.EPOCH = 1
        self.PATH = {}
        self.PATH['fig_save_folder'] = osp.join(PROJECT_ROOT, "figs/figs_dogs", self.MODEL_NAME, self.MODEL_SIZE,
                                                "reliability/g/loop{}".format(self.LOOP))
        self.PATH['fig_save_path'] = osp.join(self.PATH['fig_save_folder'], "{}.png".format(self.block_num))
        self.PATH["g_save_folder"] = osp.join(PROJECT_ROOT, "models/models_dogs", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability/g/loop{}".format(self.LOOP), str(self.block_num))
        self.PATH["ga_save_path"] = osp.join(self.PATH["g_save_folder"], "ga.pth")
        self.PATH["gb_save_path"] = osp.join(self.PATH["g_save_folder"], "gb.pth")
        self.PATH["gc_save_path"] = osp.join(self.PATH["g_save_folder"], "gc.pth")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["g_save_folder"], "data.bin.pth")
        self.PATH["student_folder"] = osp.join(PROJECT_ROOT, "models/models_dogs", self.MODEL_NAME, self.MODEL_SIZE,
                                              "reliability", str(self.block_num))
        self.PATH["current_path"] = osp.join(self.PATH["student_folder"], "current.pth")
        self.PATH["conv1_path"] = osp.join(self.PATH["student_folder"], "conv1.pth")
        self.PATH["conv2_path"] = osp.join(self.PATH["student_folder"], "conv2.pth")
        self.PATH["conv3_path"] = osp.join(self.PATH["student_folder"], "conv3.pth")
        if len(self.PREVIOUS_LIST) != 0:
            self.PATH["previous_g_folder"] = osp.join(PROJECT_ROOT, "models/models_dogs", self.MODEL_NAME, self.MODEL_SIZE,
                                                      "reliability/g/loop{}".format(self.LOOP), str(self.PREVIOUS_LIST[-1]))
            self.PATH["previous_ga_path"] = osp.join(self.PATH["previous_g_folder"], "ga.pth")
            self.PATH["previous_gb_path"] = osp.join(self.PATH["previous_g_folder"], "gb.pth")
            self.PATH["previous_gc_path"] = osp.join(self.PATH["previous_g_folder"], "gc.pth")

        if not os.path.exists(self.PATH['fig_save_folder']):
            os.makedirs(self.PATH['fig_save_folder'])
        if not os.path.exists(self.PATH["g_save_folder"]):
            os.makedirs(self.PATH["g_save_folder"])

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
        sampler_train = torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=2000)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.BATCH_SIZE, sampler=sampler_train)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DOGS_mean, DOGS_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.DATA_ROOT, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False)

    def _prepare_model(self):
        self.current = resnetdisentanglerbig(self.block_num)
        self.current.load_state_dict(torch.load(self.PATH["current_path"]))
        self.current.to(self.GPU_ID)
        print(f' - current {self.PATH["current_path"]}')
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False)).to(self.GPU_ID)
        self.conv1.load_state_dict(torch.load(self.PATH["conv1_path"]))
        self.conv2.load_state_dict(torch.load(self.PATH["conv2_path"]))
        self.conv3.load_state_dict(torch.load(self.PATH["conv3_path"]))
        print(f' - conv1   {self.PATH["conv1_path"]}')
        print(f' - conv2   {self.PATH["conv2_path"]}')
        print(f' - conv3   {self.PATH["conv3_path"]}')

        self.g_optimizer = torch.optim.Adam(
            [{'params': self.ga.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gb.parameters(), 'lr': self.G_LR_LIST[0]},
             {'params': self.gc.parameters(), 'lr': self.G_LR_LIST[0]}],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.criterion = nn.MSELoss()

    def _prepare_g(self):
        self.ga = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        self.gb = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        self.gc = nn.Sequential(nn.Conv2d(512, 512, stride=1, kernel_size=1, bias=False))
        if len(self.PREVIOUS_LIST) != 0:
            print(self.PATH['previous_g_folder'] + " loaded.")
            self.ga.load_state_dict(torch.load(self.PATH["previous_ga_path"]))
            self.gb.load_state_dict(torch.load(self.PATH["previous_gb_path"]))
            self.gc.load_state_dict(torch.load(self.PATH["previous_gc_path"]))
        self.ga.to(self.GPU_ID)
        self.gb.to(self.GPU_ID)
        self.gc.to(self.GPU_ID)


if __name__ == '__main__':
    train_list = [1, 2, 4, 8, 16, 32]
    batch_list = [64, 64, 64, 64, 32, 32]
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--rec-initial-lr", type=int, default=-3)
    parser.add_argument("--rec-final-lr", type=int, default=-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--rec-num-epochs', type=int, default=500)
    parser.add_argument('--block-num', type=int, default=None, help='train with n blocks')
    parser.add_argument('--previous-list', type=list, default=None, help='previous models')
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--g-initial-lr", type=int, default=-3)
    parser.add_argument("--g-final-lr", type=int, default=-4)
    parser.add_argument('--g-num-epochs', type=int, default=100)
    parser.add_argument('--loop', type=int, default=10)
    args = parser.parse_args()
    print(args)
    if args.dataset == "cifar10":
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            r = Rec_CIFAR10(args)
            r.prepare()
            r.run()
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            trainer = G_CIFAR10(args)
            trainer.prepare()
            trainer.run()
    elif args.dataset == "cub":
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            r = Rec_CUB(args)
            r.prepare()
            r.run()
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            trainer = G_CUB(args)
            trainer.prepare()
            trainer.run()
    elif args.dataset == "dogs":
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            r = Rec_DOGS(args)
            r.prepare()
            r.run()
        for i in range(0, len(train_list)):
            args.block_num = train_list[i]
            args.previous_list = train_list[:i]
            args.batch_size = batch_list[i]
            print(args)
            trainer = G_DOGS(args)
            trainer.prepare()
            trainer.run()

