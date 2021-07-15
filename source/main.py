import os
import sys
working_path = os.path.dirname(os.path.dirname(sys.argv[0])) + '/'  # twice dir name to get parent
print("working path: " + working_path)
os.chdir(working_path)
sys.path.append(working_path)

import numpy as np
import torch

from source import configuration_loader
from source.experiments.experiment import BasicExperiment
from source.registry.model_registry import model_dict
from source.registry.learning_rate_schedule_registry import learning_rate_schedule_dict
from source.registry.optimizer_registry import optimizer_dict
from source.registry.dataset_registry import dataset_dict
from source.tools.pickle_writer import PickleWriter
import random

config = configuration_loader.parse_configuration_file()

working_path = os.path.dirname(os.path.dirname(sys.argv[0])) + '/'  # twice dir name to get parent
print("working path: " + working_path)

# check gpus
# device_lib.list_local_devices() shows also cpus
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
num_available_gpus = torch.cuda.device_count()
# assert num_available_gpus >= config.num_gpus
print("GPUs available: {1:d}  \t GPUs used: {1:d}".format(num_available_gpus, config.num_gpus))

# Set random seed:
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
random.seed(config.random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

optimizer_func = optimizer_dict[config.optimizer]
optimizer_params = config.optimizer_args
learning_rate_schedule_func = learning_rate_schedule_dict[config.schedule]
learning_rate_schedule_args = config.schedule_args
model_func = model_dict[config.model]
dataset_params = {"dataset_path": config.dataset_path, "train_data_size": config.train_data_size,
                  "batch_size": config.batch_size, "fraction": config.dataset_fraction}

dataset_func = dataset_dict[config.dataset_name]

log_path = os.path.join(working_path, "output/")
# log_path = "/data/"
os.makedirs(log_path, exist_ok=True)

writer = PickleWriter(log_dir=os.path.join(log_path, config.experiment_name, "tb/"),
                      pickle_file_path=os.path.join(log_path, config.experiment_name, "log.pickle"))
dict = vars(config)
for name, value in dict.items():
    writer.add_text('parameters/' + name, str(value))

experiment = BasicExperiment(config.experiment_name, model_func, optimizer_func, optimizer_params,
                             learning_rate_schedule_func, learning_rate_schedule_args, dataset_func,
                             dataset_params, writer,
                             log_path=log_path, save_weights_each_step=False)

experiment.train(config.training_steps, eval_after_steps=None)  #
experiment.test()
experiment.save()

a = 2
