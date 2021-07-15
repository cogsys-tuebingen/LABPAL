from source.data_handling.dataset_loader import *


dataset_dict = {"CIFAR-10": get_cifar10_dataloaders,
                "CIFAR-10_deterministic": get_cifar10_dataloaders_deterministic,
                "CIFAR-100": get_cifar100_dataloaders,
                "MNIST": get_mnist_dataloaders,
                "ImageNet": get_imagenet_dataloaders,
                "SVHN": get_svhn_dataloaders,
                "Fashion-MNIST": get_fashion_mnist_dataloaders,
                }
