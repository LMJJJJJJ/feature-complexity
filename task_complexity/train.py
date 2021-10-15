PROJECT_ROOT = "./"
import sys
sys.path.append(PROJECT_ROOT)
sys.path.append("../")

import os
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

from tools import save_obj, load_obj, update_lr
from tasks import task
from net.resnet import resnetdisentangler


class TaskTrainer():
    def __init__(self, args):
        self.BATCH_SIZE = args.batch_size
        self.GPU_ID = args.gpu_id
        self.NUM_EPOCHS = args.num_epochs
        self.DATA_ROOT = "./data"
        self.TASK_DIFFICULTY = args.task_difficulty
        self.LR_LIST = np.logspace(args.initial_lr, args.final_lr, args.num_epochs)

        self.PATH = {}
        self.PATH["task_path"] = osp.join(PROJECT_ROOT, f"tasks/task_{self.TASK_DIFFICULTY}.pth")
        self.PATH["save_root"] = osp.join(PROJECT_ROOT, f"pretrained_models/task_{self.TASK_DIFFICULTY}")
        self.PATH['model_save_path'] = osp.join(self.PATH["save_root"], f"task_{self.TASK_DIFFICULTY}_model.pth")
        self.PATH['fig_save_path'] = osp.join(self.PATH["save_root"], f"task_{self.TASK_DIFFICULTY}_curve.png")
        self.PATH['plot_data_save_path'] = osp.join(self.PATH["save_root"], f"task_{self.TASK_DIFFICULTY}_data.bin")

        if not os.path.exists(self.PATH["save_root"]):
            os.makedirs(self.PATH["save_root"])

    def _generate_plot_dic(self):
        self.plot_dic = {
            "train_loss": [],
            "test_loss": []
        }

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

    def prepare(self):
        self._generate_plot_dic()
        self._prepare_dataset()
        self._prepare_model()

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
        self.task = task(self.TASK_DIFFICULTY).to(self.GPU_ID)
        self.task.load_state_dict(torch.load(self.PATH["task_path"]))
        print(f'{self.PATH["task_path"]} loaded.')
        self.model = resnetdisentangler(6).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()

    def train_model(self):
        self.task.eval()
        self.model.train()
        train_loss = 0
        print("Learning rate is", self.LR_LIST[self.EPOCH - 1])
        update_lr(self.optimizer, [self.LR_LIST[self.EPOCH - 1]])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images = images.to(self.GPU_ID)
            self.optimizer.zero_grad()
            output = self.model(images)
            target = self.task(images).detach()
            loss = self.criterion(output, target)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.data * images.shape[0]

        train_loss = train_loss / len(self.train_loader.sampler)
        self.plot_dic["train_loss"].append(train_loss.item())
        return train_loss

    def eval_model(self):
        self.task.eval()
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images = images.to(self.GPU_ID)
                output = self.model(images)
                target = self.task(images).detach()
                loss = self.criterion(output, target)
                test_loss += loss.data * images.shape[0]
        test_loss = test_loss / len(self.test_loader.dataset)
        self.plot_dic["test_loss"].append(test_loss)
        return test_loss

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            train_loss = self.train_model()
            print("Train Loss: {:.5f}".format(train_loss))
            test_loss = self.eval_model()
            print("Test Loss: {:.5f}".format(test_loss))
            self.draw()
            self.save()
        torch.save(self.model.to("cpu").state_dict(), self.PATH['model_save_path'])

    def save(self):
        print("Saving...")
        torch.save(self.model.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['plot_data_save_path'])


def main():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--initial-lr', type=int, default=-3)
    parser.add_argument('--final-lr', type=int, default=-5)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--task-difficulty', type=int, default=None)
    args = parser.parse_args()
    print(args)

    trainer = TaskTrainer(args)
    trainer.prepare()
    trainer.run()


if __name__ == '__main__':
    main()