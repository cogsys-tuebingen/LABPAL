import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN, FashionMNIST
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST
from sklearn.datasets import load_svmlight_file


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

    def reset_iterator(self):
        self.dataset_iterator = super().__iter__()


def get_mnist_dataloaders(dataset_path="../../datasets/mnist", train_data_size=40000, batch_size=100, shuffle=True, fraction="not implemented"):
    mean = [0.1307, ]
    std = [0.3081]
    num_classes = 10
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)
                                          ])
    # download= os.path.exists(dataset_path) and os.path.isfile(dataset_path)
    full_train_dataset = MNIST(root=os.path.join(dataset_path, "train"), train=True, transform=transform_train,
                               download=True)

    train_size = train_data_size
    test_size = len(full_train_dataset) - train_size
    lengths = [train_size, test_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))
    num_workers = 4
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                          shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    test_dataset = MNIST(root=os.path.join(dataset_path, "test"), train=False, transform=transform_test, download=True)
    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset), num_classes


def get_fashion_mnist_dataloaders(dataset_path="../../datasets/fashion_mnist", train_data_size=55000, batch_size=100,
                                  fraction=1.0, shuffle=True,):
    mean = [0.2852]
    std = [0.3197]
    # 60000 training images 10000 test images
    num_classes = 10
    transform_train = transforms.Compose([transforms.ToTensor()])
    # download= os.path.exists(dataset_path) and os.path.isfile(dataset_path)
    full_train_dataset = FashionMNIST(root=os.path.join(dataset_path, "train"), train=True, transform=transform_train,
                                      download=True)

    train_size = train_data_size
    val_size = len(full_train_dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))

    if fraction != 1.0:
        ts = int(train_size * fraction)
        vs = int(val_size * fraction)
        lengths_train = [ts, train_size - ts]
        lengths_val = [vs, val_size - vs]
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, lengths_train,
                                                                 torch.Generator().manual_seed(42))
        val_dataset, _ = torch.utils.data.dataset.random_split(val_dataset, lengths_val,
                                                               torch.Generator().manual_seed(42))

    num_workers = 4
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                          shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    test_dataset = FashionMNIST(root=os.path.join(dataset_path, "test"), train=False, transform=transform_test,
                                download=True)
    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset) // batch_size * batch_size, num_classes


def get_cifar10_dataloaders(dataset_path="../../datasets/cifar10_dataset", train_data_size=40000, batch_size=100,
                            fraction=1.0, shuffle=True, augment=True):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    num_classes = 10
    if augment:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # download= os.path.exists(dataset_path) and os.path.isfile(dataset_path)
    full_train_dataset = CIFAR10(root=os.path.join(dataset_path, "train"), train=True, transform=transform_train,
                                 download=True)

    train_size = train_data_size
    val_size = len(full_train_dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))
    if fraction != 1.0:
        ts = int(train_size * fraction)
        vs = int(val_size * fraction)
        lengths_train = [ts, train_size - ts]
        lengths_val = [vs, val_size - vs]
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, lengths_train,
                                                                 torch.Generator().manual_seed(42))
        val_dataset, _ = torch.utils.data.dataset.random_split(val_dataset, lengths_val,
                                                               torch.Generator().manual_seed(42))

    num_workers = 4
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                          shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    test_dataset = CIFAR10(root=os.path.join(dataset_path, "test"), train=False, transform=transform_test,
                           download=True)
    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset), num_classes


def get_cifar10_dataloaders_deterministic(dataset_path="../../datasets/cifar10_dataset", train_data_size=40000,
                                          batch_size=100, fraction=1.0, shuffle=False, augment=False):
    return get_cifar10_dataloaders(dataset_path, train_data_size, batch_size, fraction, shuffle, augment)


def get_cifar100_dataloaders(dataset_path="../../datasets/cifar100_dataset", train_data_size=40000, batch_size=100,
                             num_gpus=1, fraction=1.0, shuffle=True):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=8),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    full_train_dataset = CIFAR100(root=os.path.join(dataset_path, "train"), train=True, transform=transform_train,
                                  download=True)

    train_size = train_data_size
    val_size = len(full_train_dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))

    if fraction != 1.0:
        ts = int(train_size * fraction)
        vs = int(val_size * fraction)
        lengths_train = [ts, train_size - ts]
        lengths_val = [vs, val_size - vs]
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, lengths_train,
                                                                 torch.Generator().manual_seed(42))
        val_dataset, _ = torch.utils.data.dataset.random_split(val_dataset, lengths_val,
                                                               torch.Generator().manual_seed(42))

    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    test_dataset = CIFAR100(root=os.path.join(dataset_path, "test"), train=False, transform=transform_test,
                            download=True)

    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, )
    num_classes = 100
    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset), num_classes


def get_svhn_dataloaders(dataset_path="../../datasets/svhn", train_data_size=65000, batch_size=100, fraction=1.0,
                         shuffle=True):
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]
    # 73257 samples
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    full_train_dataset = SVHN(root=os.path.join(dataset_path, "train"), split='train', transform=transform_train,
                              download=True)

    train_size = train_data_size
    val_size = len(full_train_dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))

    if fraction != 1.0:
        ts = int(train_size * fraction)
        vs = int(val_size * fraction)
        lengths_train = [ts, train_size - ts]
        lengths_val = [vs, val_size - vs]
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, lengths_train,
                                                                 torch.Generator().manual_seed(42))
        val_dataset, _ = torch.utils.data.dataset.random_split(val_dataset, lengths_val,
                                                               torch.Generator().manual_seed(42))

    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    test_dataset = SVHN(root=os.path.join(dataset_path, "test"), split='train', transform=transform_test, download=True)

    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, )
    num_classes = 10
    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset), num_classes


def get_imagenet_dataloaders(dataset_path="../../datasets/ImageNet_ILSVRC2012/", train_data_size=1080000,
                             batch_size=100, fraction=1.0, shuffle=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 1000
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    # download= os.path.exists(dataset_path) and os.path.isfile(dataset_path)

    full_train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform_train)
    train_size = train_data_size
    val_size = len(full_train_dataset) - train_size
    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
                                                                       torch.Generator().manual_seed(42))

    if fraction != 1.0:
        ts = int(train_size * fraction)
        vs = int(val_size * fraction)
        lengths_train = [ts, train_size - ts]
        lengths_val = [vs, val_size - vs]
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset, lengths_train,
                                                                 torch.Generator().manual_seed(42))
        val_dataset, _ = torch.utils.data.dataset.random_split(val_dataset, lengths_val,
                                                               torch.Generator().manual_seed(42))
    num_workers = 4
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                          shuffle=shuffle,
                                          drop_last=True,
                                          pin_memory=True)

    val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                        drop_last=True,
                                        pin_memory=True)

    transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    test_dataset = ImageFolder(root=os.path.join(dataset_path, "val"), transform=transform_test)
    test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset) // batch_size * batch_size, len(
        val_dataset) // batch_size * batch_size, len(
        test_dataset), num_classes


#import numpy as np

# def get_mushroom_dataloaders(dataset_path="../../datasets/mushrooms/mushrooms.txt", train_data_size=40000, batch_size=100):
#
#     sigma=0.5
#
#     X, y = load_svmlight_file(dataset_path)
#
#     labels = np.unique(y)
#
#     y[y == labels[0]] = 0
#     y[y == labels[1]] = 1
#     # TODO: (amishkin) splits = train_test_split(X, y, test_size=0.2, shuffle=shuffle, random_state=9513451)
#     splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
#     X_train, X_test, Y_train, Y_test = splits
#
#
#
#
#
#     mean = [0.4914, 0.4822, 0.4465]
#     std = [0.2023, 0.1994, 0.2010]
#     num_classes = 10
#     transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(mean, std)])
#     # download= os.path.exists(dataset_path) and os.path.isfile(dataset_path)
#     full_train_dataset = CIFAR10(root=os.path.join(dataset_path, "train"), train=True, transform=transform_train,
#                                  download=True)
#
#     train_size = train_data_size
#     test_size = len(full_train_dataset) - train_size
#     lengths = [train_size, test_size]
#     train_dataset, val_dataset = torch.utils.data.dataset.random_split(full_train_dataset, lengths,
#                                                                        torch.Generator().manual_seed(42))
#     num_workers = 4
#     train_dataloader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
#                                           drop_last=True,
#                                           pin_memory=True)
#
#     val_dataloader = InfiniteDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
#                                         drop_last=True,
#                                         pin_memory=True)
#
#     transform_test = transforms.Compose([transforms.ToTensor(),
#                                          transforms.Normalize(mean, std)])
#     test_dataset = CIFAR10(root=os.path.join(dataset_path, "test"), train=False, transform=transform_test,
#                            download=True)
#     test_dataloader = InfiniteDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
#
#     return train_dataloader, val_dataloader, test_dataloader, len(train_dataset), len(val_dataset), len(
#         test_dataset), num_classes


