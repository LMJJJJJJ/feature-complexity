import argparse
import sys
import os
import os.path as osp
from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional
from PIL import Image
import xml.etree.ElementTree
import scipy
import scipy.io

sys.path.append("..")
from tools import generate_dataset_root, check_data_size


def generate_cifar10(data_root, data_size):
    num_class = 10
    num_per_type = data_size // num_class
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root, train=True, download=True, transform=transform),
        batch_size=1,
        shuffle=True,
        num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root, train=False, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    if not os.path.exists(osp.join(data_root, f"CIFAR10{data_size}/train")):
        os.makedirs(osp.join(data_root, f"CIFAR10{data_size}/train"))
    if not os.path.exists(osp.join(data_root, f"CIFAR10{data_size}/test")):
        os.makedirs(osp.join(data_root, f"CIFAR10{data_size}/test"))
    if not os.path.exists(osp.join(data_root, f"CIFAR10{data_size}/label")):
        os.makedirs(osp.join(data_root, f"CIFAR10{data_size}/label"))
    train_slot = [num_per_type for _ in range(num_class)]
    train_count = 1

    for i, (image, label) in tqdm(enumerate(train_loader)):
        filename = "0%05d" % train_count
        if train_slot[label.item()] > 0:
            train_slot[label.item()] -= 1
            torch.save(image[0], osp.join(data_root, f"CIFAR10{data_size}/train", "%s.pt" % filename))
            torch.save(label[0], osp.join(data_root, f"CIFAR10{data_size}/label", "%s.pt" % filename))
            train_count += 1
        if sum(train_slot) == 0:
            break

    for i, (image, label) in tqdm(enumerate(test_loader)):
        filename = "1%05d" % (i + 1)
        torch.save(image[0], osp.join(data_root, f"CIFAR10{data_size}/test", "%s.pt" % filename))
        torch.save(label[0], osp.join(data_root, f"CIFAR10{data_size}/label", "%s.pt" % filename))


def generate_cub(data_root, data_size=None):
    num_class = 200
    if data_size is None:
        num_per_class = 99999
        data_size = 'strong'
    else:
        num_per_class = data_size // num_class
    slot = {i: num_per_class for i in range(1, 201)}
    src_path = "../data/CUB/CUB_200_2011"
    dst_path = osp.join(data_root, "CUB_{}".format(data_size))
    # create folders
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if not os.path.exists(osp.join(dst_path, "train")):
        os.makedirs(osp.join(dst_path, "train"))
        os.makedirs(osp.join(dst_path, "test"))
        for class_dir in os.listdir(osp.join(src_path, 'images')):
            os.makedirs(osp.join(dst_path, 'train', class_dir))
            os.makedirs(osp.join(dst_path, 'test', class_dir))

    with open(osp.join(src_path, 'images.txt'), "r") as images_f:
        img_path = images_f.readlines()
        img_path = [line.strip().split() for line in img_path]
        img_path = {int(item[0]): item[1] for item in img_path}

    with open(osp.join(src_path, 'image_class_labels.txt'), "r") as image_class_labels_f:
        img_class = image_class_labels_f.readlines()
        img_class = [line.strip().split() for line in img_class]
        img_class = {int(item[0]): int(item[1]) for item in img_class}

    with open(osp.join(src_path, 'train_test_split.txt'), "r") as train_test_split_f:
        img_istrain = train_test_split_f.readlines()
        img_istrain = [line.strip().split() for line in img_istrain]
        img_istrain = {int(item[0]): int(item[1]) for item in img_istrain}

    with open(osp.join(src_path, 'bounding_boxes.txt'), "r") as bounding_boxes_f:
        bounding_boxes = bounding_boxes_f.readlines()
        bounding_boxes = [line.strip().split() for line in bounding_boxes]
        bounding_boxes = [[int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4])]
                          for item in bounding_boxes]
        bounding_boxes = {item[0]:item[1:] for item in bounding_boxes}

    for img, path in tqdm(img_path.items(), desc="Preprocessing", mininterval=1):
        istrain = img_istrain[img]
        x, y, w, h = bounding_boxes[img]
        cls = img_class[img]

        if istrain == 0:
            pass
        elif istrain == 1 and slot[cls] > 0:
            slot[cls] -= 1
        else:
            continue

        with Image.open(osp.join(src_path, "images", path)) as im:
            im = transforms.functional.resized_crop(im, y, x, h, w, [224, 224])
            if istrain == 1:
                im.save(osp.join(dst_path, "train", path))
            else:
                im.save(osp.join(dst_path, "test", path))


def parse_dog_bndbox(path):
    e = xml.etree.ElementTree.parse(path).getroot()
    boxes = []
    for objs in e.iter('object'):
        xmin = int(objs.find('bndbox').find('xmin').text)
        ymin = int(objs.find('bndbox').find('ymin').text)
        xmax = int(objs.find('bndbox').find('xmax').text)
        ymax = int(objs.find('bndbox').find('ymax').text)
        width = xmax - xmin
        height = ymax - ymin
        boxes.append([xmin, ymin, width, height])
    return boxes


def generate_dogs(data_root, data_size=None):
    num_class = 120
    if data_size is None:
        num_per_class = 99999
        data_size = 'strong'
    else:
        num_per_class = data_size // num_class
    src_path = "../data/DOGS/StanfordDogs"
    dst_path = osp.join(data_root, "DOGS_{}".format(data_size))
    all_classes = os.listdir(osp.join(src_path, "Images"))
    slot = {c: num_per_class for c in all_classes}
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if not os.path.exists(osp.join(dst_path, "train")):
        os.makedirs(osp.join(dst_path, "train"))
        os.makedirs(osp.join(dst_path, "test"))
        for class_dir in all_classes:
            os.makedirs(osp.join(dst_path, "train", class_dir))
            os.makedirs(osp.join(dst_path, "test", class_dir))

    train_img_path = scipy.io.loadmat(osp.join(src_path, 'train_list.mat'))['file_list']
    train_img_path = [i.item()[0] for i in train_img_path]
    train_annotation_path = scipy.io.loadmat(osp.join(src_path, 'train_list.mat'))['annotation_list']
    train_annotation_path = [i.item()[0] for i in train_annotation_path]

    test_img_path = scipy.io.loadmat(osp.join(src_path, 'test_list.mat'))['file_list']
    test_img_path = [i.item()[0] for i in test_img_path]
    test_annotation_path = scipy.io.loadmat(osp.join(src_path, 'test_list.mat'))['annotation_list']
    test_annotation_path = [i.item()[0] for i in test_annotation_path]

    # Saving train set
    for img_path, anno_path in tqdm(zip(train_img_path, train_annotation_path), desc="Train Set", mininterval=1):
        assert img_path[:-4] == anno_path
        boxes = parse_dog_bndbox(osp.join(src_path, "Annotation", anno_path))
        cls = anno_path.split("/")[0]
        if slot[cls] == 0:
            continue
        slot[cls] -= 1
        with Image.open(osp.join(src_path, "Images", img_path)) as im:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                im_crop = transforms.functional.resized_crop(im, y, x, h, w, [224, 224])
                if i == 0:
                    try:
                        im_crop.save(osp.join(dst_path, "train", img_path))
                    except:
                        print("Converting {} to RGB".format(img_path))
                        im_crop = im_crop.convert('RGB')
                        im_crop.save(osp.join(dst_path, "train", img_path))
                else:
                    try:
                        im_crop.save(osp.join(dst_path, "train", anno_path+"_{}.jpg".format(i)))
                    except:
                        print("Converting {} to RGB".format(img_path))
                        im_crop = im_crop.convert('RGB')
                        im_crop.save(osp.join(dst_path, "train", anno_path+"_{}.jpg".format(i)))


    # Saving test set
    for img_path, anno_path in tqdm(zip(test_img_path, test_annotation_path), desc="Test Set", mininterval=1):
        assert img_path[:-4] == anno_path
        boxes = parse_dog_bndbox(osp.join(src_path, "Annotation", anno_path))
        with Image.open(osp.join(src_path, "Images", img_path)) as im:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                im_crop = transforms.functional.resized_crop(im, y, x, h, w, [224, 224])
                if i == 0:
                    try:
                        im_crop.save(osp.join(dst_path, "test", img_path))
                    except:
                        print("Converting {} to RGB".format(img_path))
                        im_crop = im_crop.convert('RGB')
                        im_crop.save(osp.join(dst_path, "test", img_path))
                else:
                    try:
                        im_crop.save(osp.join(dst_path, "test", anno_path+"_{}.jpg".format(i)))
                    except:
                        print("Converting {} to RGB".format(img_path))
                        im_crop = im_crop.convert('RGB')
                        im_crop.save(osp.join(dst_path, "test", anno_path+"_{}.jpg".format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--save-root', type=str, default=None)
    args = parser.parse_args()
    # check the parameters
    if args.dataset is None:
        raise Exception("Dataset not specified")
    if args.save_root is None:
        args.save_root = generate_dataset_root(args.dataset)
    check_data_size(args.dataset, args.data_size)
    # print the parameters out
    print(args)
    # generate dataset
    if args.dataset == 'cifar10':
        generate_cifar10(args.save_root, args.data_size)
    elif args.dataset == 'cub':
        generate_cub(args.save_root, args.data_size)
    elif args.dataset == 'dogs':
        generate_dogs(args.save_root, args.data_size)

