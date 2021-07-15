import torch.optim.lr_scheduler as s

learning_rate_schedule_dict = {"": None, "MultiStepLR": s.MultiStepLR,
                               "ExponentialLR": s.ExponentialLR}
