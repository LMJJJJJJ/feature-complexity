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


class DisentanglerTrainer(object):
    def __init__(self):
        self.current = None
        self.out_conv = None
        self.pretrained_model = None
        self.optimizer = None
        self.criterion = None
        self.label_criterion = None
        self.train_loader = None
        self.test_loader = None
        self.GPU_ID = None
        self.BATCH_SIZE = None
        self.EPOCH = None
        self.NUM_EPOCHS = None
        self.PATH = None
        self.LR_LIST = None

    def prepare(self):
        self._prepare_model()
        self._prepare_dataset()
        self._generate_plot_dic()

    def _generate_plot_dic(self):
        self.plot_dic = {
            "train_acc": [],
            "test_acc": [],
            "distill_loss": [],
            "label_loss_train_pretrained": [],
            "label_loss_train": [],
            "label_loss_test_pretrained": [],
            "label_loss_test": []
        }

    def _prepare_dataset(self):
        raise NotImplementedError

    def _prepare_model(self):
        raise NotImplementedError

    def train_epoch(self):
        self.current.train()
        self.out_conv.eval()
        self.pretrained_model.eval()
        step = 0
        distill_loss = 0
        label_loss_pretrained = 0
        label_loss_preconv = 0
        correct_preconv = 0
        pretrained_correct = 0
        update_lr(self.optimizer, [self.LR_LIST[self.EPOCH - 1]])
        print("Learning rate is", self.optimizer.param_groups[0]["lr"])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)
            self.optimizer.zero_grad()
            pretrained = self.pretrained_model(images).detach()
            output = self.current(images)

            loss = self.criterion(output, pretrained)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            distill_loss += loss.data

            output = output.detach()
            output_preconv = self.out_conv(output)
            y_pred_preconv = output_preconv.data.max(1)[1]
            correct_preconv += y_pred_preconv.eq(labels.data).sum()
            label_loss_preconv += self.label_criterion(output_preconv, labels)

            labels_pretrained = self.out_conv(F.relu(pretrained)).detach()
            label_loss_pretrained += self.label_criterion(labels_pretrained, labels).data
            pretrained_correct += labels_pretrained.data.max(1)[1].eq(labels.data).sum()

            step += 1
            if step % 100 == 0:
                print(' distillation loss: {:.4f}'.format(loss.item()))

        acc = 100. * float(correct_preconv) / len(self.train_loader.sampler)
        pretrained_acc = 100. * float(pretrained_correct) / len(self.train_loader.sampler)
        length = len(self.train_loader.sampler) // self.BATCH_SIZE # revised
        distill_loss = distill_loss / length
        label_loss_pretrained = label_loss_pretrained / length
        label_loss_preconv = label_loss_preconv / length

        self.plot_dic['distill_loss'].append(distill_loss.item())
        self.plot_dic['train_acc'].append(acc)
        self.plot_dic['label_loss_train_pretrained'].append(label_loss_pretrained.item())
        self.plot_dic['label_loss_train'].append(label_loss_preconv.item())

        print("Train Set: distillation Loss (feature): {0:2.3f}".format(distill_loss))
        print("Acc (add pretrained outconv): {0:.3f}%".format(acc))
        print("Train Set: pretrained accuracy: {:.3f}%".format(pretrained_acc))

    def eval_epoch(self):
        self.current.eval()
        self.out_conv.eval()
        self.pretrained_model.eval()
        distill_correct_preconv = 0
        pretrained_correct = 0
        label_loss_pretrained = 0
        label_loss_preconv = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images, labels = images.to(self.GPU_ID), labels.to(self.GPU_ID)

                output = self.current(images)

                prediction_preconv = self.out_conv(output)
                distill_correct_preconv += prediction_preconv.data.max(1)[1].eq(labels.data).sum()

                pretrained_prediction = self.out_conv(F.relu(self.pretrained_model(images)))
                pretrained_correct += pretrained_prediction.data.max(1)[1].eq(labels.data).sum()

                label_loss_pretrained += self.label_criterion(pretrained_prediction, labels)
                label_loss_preconv += self.label_criterion(prediction_preconv, labels)

        distill_acc_preconv = 100. * float(distill_correct_preconv) / len(self.test_loader.dataset)
        pretrained_acc = 100. * float(pretrained_correct) / len(self.test_loader.dataset)
        length = len(self.test_loader.dataset) // self.BATCH_SIZE
        label_loss_pretrained = label_loss_pretrained / length
        label_loss_preconv = label_loss_preconv / length
        self.plot_dic['test_acc'].append(distill_acc_preconv)
        self.plot_dic['label_loss_test_pretrained'].append(label_loss_pretrained.item())
        self.plot_dic['label_loss_test'].append(label_loss_preconv.item())
        print('Accuracy of the network on the test images (add pretrained outconv): {} %'.format(distill_acc_preconv))
        print('Accuracy of the network on the test images (pretrained model): {} %'.format(pretrained_acc))

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            self.train_epoch()
            self.eval_epoch()
            self.save()
            self.draw()
        torch.save(self.current.to("cpu").state_dict(), self.PATH['model_save_path'])

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
        # distillation loss (MSE) on training set
        plt.subplot(2, 2, 2)
        x = range(1, len(self.plot_dic["distill_loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["distill_loss"], label="distill_loss")
        # label loss (CrossEntropy) on training set
        plt.subplot(2, 2, 3)
        x = range(1, len(self.plot_dic["label_loss_train_pretrained"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["label_loss_train_pretrained"], label="label_loss_train_pretrained")
        plt.plot(x, self.plot_dic["label_loss_train"], label="label_loss_train")
        plt.legend()
        # label loss (CrossEntropy) on testing set
        plt.subplot(2, 2, 4)
        x = range(1, len(self.plot_dic["label_loss_test_pretrained"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["label_loss_test_pretrained"], label="label_loss_test_pretrained")
        plt.plot(x, self.plot_dic["label_loss_test"], label="label_loss_test")
        plt.legend()
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        torch.save(self.current.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


class Trainer_CIFAR10(DisentanglerTrainer):
    def __init__(self, args):
        super(Trainer_CIFAR10, self).__init__()
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.BLOCK_NUM = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.NUM_EPOCHS = args.num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = './data/CIFAR10'
        self.EPOCH = 1
        self.PATH = {}
        self.PATH["fig_save_folder"] = osp.join(PROJECT_ROOT, f"figs/figs_cifar10/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH["model_save_folder"] = osp.join(PROJECT_ROOT, f"models/models_cifar10/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH['fig_save_path'] = osp.join(self.PATH["fig_save_folder"], "{}_curve.png".format(self.BLOCK_NUM))
        self.PATH['model_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.BLOCK_NUM))
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_data.bin".format(self.BLOCK_NUM))
        self.PATH['pretrained_net'] = osp.join(PROJECT_ROOT,
                                               f"pretrained_models/{self.MODEL_NAME}_cifar10/{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        if len(self.PREVIOUS_LIST) == 0:
            self.PATH['previous_model'] = None
        else:
            self.PATH['previous_model'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

        if not os.path.exists(self.PATH["fig_save_folder"]):
            os.makedirs(self.PATH["fig_save_folder"])
        if not os.path.exists(self.PATH["model_save_folder"]):
            os.makedirs(self.PATH["model_save_folder"])

    def _prepare_dataset(self):
        print(self.DATA_ROOT)
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
        if self.MODEL_NAME == 'resnet8':
            from net.resnet import resnet8
            self.pretrained_model = resnet8(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet14':
            from net.resnet import resnet14
            self.pretrained_model = resnet14(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet20':
            from net.resnet import resnet20
            self.pretrained_model = resnet20(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet32':
            from net.resnet import resnet32
            self.pretrained_model = resnet32(disentangle=True).to(self.GPU_ID)
        elif self.MODEL_NAME == 'resnet44':
            from net.resnet import resnet44
            self.pretrained_model = resnet44(disentangle=True).to(self.GPU_ID)
        self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
        self.out_conv = nn.Sequential(
            nn.AvgPool2d(8, 8),
            Flatten(),
            self.pretrained_model.linear
        )

        from net.resnet import resnetdisentangler
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentangler(self.BLOCK_NUM).to(self.GPU_ID)
        else:
            self.current = get_current(self.BLOCK_NUM, self.PATH['previous_model']).to(self.GPU_ID)
            print("{} loaded".format(self.PATH['previous_model']))
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.LR_LIST[0], betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


class Trainer_CUB(DisentanglerTrainer):
    def __init__(self, args):
        super(Trainer_CUB, self).__init__()
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.BLOCK_NUM = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.NUM_EPOCHS = args.num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = './data/CUB/CUB_strong/'
        self.EPOCH = 1
        self.PATH = {}
        self.PATH["fig_save_folder"] = osp.join(PROJECT_ROOT, f"figs/figs_cub/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH["model_save_folder"] = osp.join(PROJECT_ROOT, f"models/models_cub/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH['fig_save_path'] = osp.join(self.PATH["fig_save_folder"], "{}_curve.png".format(self.BLOCK_NUM))
        self.PATH['model_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.BLOCK_NUM))
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_data.bin".format(self.BLOCK_NUM))
        self.PATH['pretrained_net'] = osp.join(PROJECT_ROOT,
                                               f"pretrained_models/{self.MODEL_NAME}_cub/{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        if len(self.PREVIOUS_LIST) == 0:
            self.PATH['previous_model'] = None
        else:
            self.PATH['previous_model'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

        if not os.path.exists(self.PATH["fig_save_folder"]):
            os.makedirs(self.PATH["fig_save_folder"])
        if not os.path.exists(self.PATH["model_save_folder"]):
            os.makedirs(self.PATH["model_save_folder"])

    def _prepare_dataset(self):
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
        if self.MODEL_NAME == "resnet18":
            from net.resnetbig import resnet18_cub
            self.pretrained_model = resnet18_cub(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.fc
            )
        elif self.MODEL_NAME == "resnet34":
            from net.resnetbig import resnet34_cub
            self.pretrained_model = resnet34_cub(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.fc
            )
        elif self.MODEL_NAME == "vgg16":
            from net.vggbig import vgg16_cub
            self.pretrained_model = vgg16_cub(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.classifier
            )

        from net.resnetbig import resnetdisentanglerbig
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentanglerbig(self.BLOCK_NUM).to(self.GPU_ID)
        else:
            self.current = get_current_big(self.BLOCK_NUM, self.PATH['previous_model']).to(self.GPU_ID)
            print("{} loaded".format(self.PATH['previous_model']))
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.LR_LIST[0], betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


class Trainer_DOGS(DisentanglerTrainer):
    def __init__(self, args):
        super(Trainer_DOGS, self).__init__()
        self.GPU_ID = args.gpu_id
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)
        self.BLOCK_NUM = args.block_num
        self.BATCH_SIZE = args.batch_size
        self.NUM_EPOCHS = args.num_epochs
        self.PREVIOUS_LIST = args.previous_list
        self.MODEL_NAME = args.model_name
        self.DATA_SIZE = args.data_size
        self.DATA_ROOT = './data/DOGS/DOGS_strong/'
        self.EPOCH = 1
        self.PATH = {}
        self.PATH["fig_save_folder"] = osp.join(PROJECT_ROOT, f"figs/figs_dogs/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH["model_save_folder"] = osp.join(PROJECT_ROOT, f"models/models_dogs/{self.MODEL_NAME}/{self.MODEL_NAME}_{self.DATA_SIZE}/disentangle")
        self.PATH['fig_save_path'] = osp.join(self.PATH["fig_save_folder"], "{}_curve.png".format(self.BLOCK_NUM))
        self.PATH['model_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.BLOCK_NUM))
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["model_save_folder"], "{}_data.bin".format(self.BLOCK_NUM))
        self.PATH['pretrained_net'] = osp.join(PROJECT_ROOT,
                                               f"pretrained_models/{self.MODEL_NAME}_dogs/{self.MODEL_NAME}_{self.DATA_SIZE}_model.pth")
        if len(self.PREVIOUS_LIST) == 0:
            self.PATH['previous_model'] = None
        else:
            self.PATH['previous_model'] = osp.join(self.PATH["model_save_folder"], "{}_model.pth".format(self.PREVIOUS_LIST[-1]))

        if not os.path.exists(self.PATH["fig_save_folder"]):
            os.makedirs(self.PATH["fig_save_folder"])
        if not os.path.exists(self.PATH["model_save_folder"]):
            os.makedirs(self.PATH["model_save_folder"])

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
        if self.MODEL_NAME == "resnet18":
            from net.resnetbig import resnet18_dogs
            self.pretrained_model = resnet18_dogs(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.fc
            )
        elif self.MODEL_NAME == "resnet34":
            from net.resnetbig import resnet34_dogs
            self.pretrained_model = resnet34_dogs(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.fc
            )
        elif self.MODEL_NAME == "vgg16":
            from net.vggbig import vgg16_dogs
            self.pretrained_model = vgg16_dogs(disentangle=True).to(self.GPU_ID)
            self.pretrained_model.load_state_dict(torch.load(self.PATH['pretrained_net']))
            print("{} loaded".format(self.PATH['pretrained_net']))
            self.out_conv = nn.Sequential(
                self.pretrained_model.avgpool,
                Flatten(),
                self.pretrained_model.classifier
            )

        from net.resnetbig import resnetdisentanglerbig
        if len(self.PREVIOUS_LIST) == 0:
            self.current = resnetdisentanglerbig(self.BLOCK_NUM).to(self.GPU_ID)
        else:
            self.current = get_current_big(self.BLOCK_NUM, self.PATH['previous_model']).to(self.GPU_ID)
            print("{} loaded".format(self.PATH['previous_model']))
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=self.LR_LIST[0], betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.label_criterion = nn.CrossEntropyLoss().to(self.GPU_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--initial-lr", type=int, default=-3)
    parser.add_argument("--final-lr", type=int, default=-5)
    parser.add_argument("--bs-list", type=list, default=[128, 128, 128, 64, 64, 32])
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument("--block-list", type=list, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--block-num', type=int, default=None, help='train with n blocks')
    parser.add_argument('--previous-list', type=list, default=None, help='previous models')
    args = parser.parse_args()
    print(args)
    if args.dataset == 'cifar10':
        for i in range(0, len(args.block_list)):
            args.block_num = args.block_list[i]
            args.previous_list = args.block_list[:i]
            args.batch_size = args.bs_list[i]
            print(args)
            trainer = Trainer_CIFAR10(args)
            trainer.prepare()
            trainer.run()
    if args.dataset == "cub":
        for i in range(0, len(args.block_list)):
            args.block_num = args.block_list[i]
            args.previous_list = args.block_list[:i]
            args.batch_size = args.bs_list[i]
            print(args)
            trainer = Trainer_CUB(args)
            trainer.prepare()
            trainer.run()
    if args.dataset == "dogs":
        for i in range(0, len(args.block_list)):
            args.block_num = args.block_list[i]
            args.previous_list = args.block_list[:i]
            args.batch_size = args.bs_list[i]
            print(args)
            trainer = Trainer_DOGS(args)
            trainer.prepare()
            trainer.run()
