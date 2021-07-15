from source.models.resnet import *
from source.models.resnet_bn import *
from source.models.vgg import *
from source.models.mobilenetv2 import *
from source.models.inception import *
from source.models.googlenet import *
from source.models.densenet import *
from source.models.mnist_nets import *

import torchvision.models as pymodels

device="cpu"

model_dict = {
    # mnist
    "mnist_conv_net": MNIST_CONV_NET,
    "mnist_fc_net": MNIST_FC_NET,

    # cifar10  models need device parameter

    "res_net_20_cifar": resnet20,
    "res_net_32_cifar": resnet32,

    "res_net_18_cifar": resnet18,
    "res_net_34_cifar": resnet34,
    "res_net_50_cifar": resnet50,
    "res_net_101_cifar": resnet101,
    "res_net_152_cifar": resnet152,

    "res_next_50_cifar": resnext50_32x4d,

    "vgg11_cifar": vgg11,
    "vgg11_bn_cifar": vgg11_bn,
    "vgg13_cifar": vgg13,
    "vgg13_bn_cifar": vgg13_bn,
    "vgg16_cifar": vgg16,
    "vgg16_bn_cifar": vgg16_bn,
    "vgg19_cifar": vgg19,
    "vgg19_bn_cifar": vgg19_bn,

    "mobilenet_v2_cifar": mobilenet_v2,

    "inception_v3_cifar": inception_v3,

    "googlenet_cifar": googlenet,

    "densenet121_cifar": densenet121,
    "densenet169_cifar": densenet169,
    "densenet201_cifar": densenet201,
    "densenet161_cifar": densenet161,

    # imagenet models do not include device parameter

    "res_net_18_imagenet": pymodels.resnet18,
    "res_net_34_imagenet": pymodels.resnet34,
    "res_net_50_imagenet": pymodels.resnet50,
    "res_net_101_imagenet": pymodels.resnet101,
    "res_net_152_imagenet": pymodels.resnet152,

    "vgg11_imagenet": pymodels.vgg11,
    "vgg11_bn_imagenet": pymodels.vgg11_bn,
    "vgg13_imagenet": pymodels.vgg13,
    "vgg13_bn_imagenet": pymodels.vgg13_bn,
    "vgg16_imagenet": pymodels.vgg16,
    "vgg16_bn_imagenet": pymodels.vgg16_bn,
    "vgg19_imagenet": pymodels.vgg19,
    "vgg19_bn_imagenet": pymodels.vgg19_bn,

    "mobilenet_v2_imagenet": pymodels.mobilenet_v2,
    #"mobilenet_v3_small_imagenet": lambda : torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=False)
    #"mobilenet_v3_small_imagenet": pymodels.mobilenet_v3_small,
       # direct method call  only works with torch >= 1.8

    "inception_v3_imagenet": pymodels.inception_v3,
    "googlenet_imagenet": pymodels.googlenet,




    "densenet121_imagenet": pymodels.densenet121,
    "densenet169_imagenet": pymodels.densenet169,
    "densenet201_imagenet": pymodels.densenet201,
    "densenet161_imagenet": pymodels.densenet161,

}
#'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161'
# VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#    'vgg19_bn', 'vgg19

