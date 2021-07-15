import torch
import pickle
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class PickleWriter(SummaryWriter):

    def __init__(self, log_dir, pickle_file_path):
        super().__init__(log_dir)
        self.pickle_file_path = pickle_file_path
        parent_dir = Path(pickle_file_path).parent
        parent_dir.mkdir(exist_ok=True)
        self.log_dict = {}


    def add_scalar(self, name, data, step=None, description=None, tensorboard=True):

            if tensorboard == True:
                super().add_scalar(name, data, step, description)
            if isinstance(data,torch.Tensor):
                data= data.item()
            #self._python_scalar(name,data,step)

    def _python_scalar(self, name, data,step):
        if name in self.log_dict:
            self.log_dict[name].append((data,step))
        else:
            self.log_dict[name] = [(data,step)]

    def flush_to_pickle(self):
        with  open(self.pickle_file_path, "wb") as output_file:
            pickle.dump(self.log_dict, output_file)
        time.sleep(600)
