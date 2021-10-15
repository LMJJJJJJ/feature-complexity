import torch
import pickle
import os
import os.path as osp


class AverageValueMeter(object):
    def __init__(self, gpu_id, dim=4096):
        self.count = 0
        self.sum = torch.zeros(dim).to(gpu_id)
        self.squared_sum = torch.zeros(dim).to(gpu_id)

    def add(self, x):
        cnt = x.shape[0]
        self.count += cnt
        for i in range(0, cnt):
            self.sum += x[i].view(-1)
            self.squared_sum += x[i].view(-1) * x[i].view(-1)

    def get_var(self):
        if self.count == 0:
            raise Exception("No data!")
        return (self.squared_sum / self.count - (self.sum / self.count) * (self.sum / self.count)).mean()

    def get_std(self):
        return torch.sqrt(self.get_var())


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def update_lr(optimizer, lr):
    for ix, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr[0]


def generate_dataset_root(dataset):
    if dataset == "cifar10":
        return '../data/CIFAR10'
    elif dataset == 'cub':
        return '../data/CUB'
    elif dataset == 'dogs':
        return '../data/DOGS'
    else:
        raise Exception(f'Unknown dataset [{dataset}]. Please choose from cifar10/cub/dogs.')


def check_data_size(dataset, data_size):
    if data_size is None:
        return
    if dataset == 'cifar10' and data_size in [200, 500, 1000, 2000, 5000]:
        return
    if dataset == 'cub' and data_size in [2000, 3000, 4000, 5000]:
        return
    if dataset == 'dogs' and data_size in [1200, 2400, 3600, 4800]:
        return
    raise Exception(f"Invalid dataset: [{dataset}-{data_size}].\n"
                    f"Please choose from\n"
                    f" * cifar10-200/500/1000/2000/5000\n"
                    f" * cub-2000/3000/4000/5000\n"
                    f" * dogs-1200/2400/3600/4800")

if __name__ == '__main__':
    check_data_size('cifar10', 200)