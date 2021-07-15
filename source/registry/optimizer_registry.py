import torch.optim as opt
from source.optimizers.pal_optimizer import PalOptimizer
from source.optimizers.sls_per_step import SLS
from source.optimizers.pls import PLS
from source.optimizers.golsi import GOLSI
from source.optimizers.lab_pal import LabPal

optimizer_dict = {
    "SGD": opt.SGD,
    "LABPAL": LabPal,
    "PAL": PalOptimizer,
    "ADAM": opt.Adam,
    "RMSPROP": opt.RMSprop,
    "ADAGRAD": opt.Adagrad,
    "ADADELTA": opt.Adadelta,
    "SLS": SLS,
    "PLS": PLS,
    "GOLSI": GOLSI,
}
