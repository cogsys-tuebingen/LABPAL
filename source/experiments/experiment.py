import inspect
import os
import time

import numpy as np
import torch

from source.optimizers.golsi import GOLSI
from source.optimizers.lab_pal import LabPal
from source.optimizers.pal_optimizer import PalOptimizer
from source.optimizers.pls import PLS
from source.optimizers.sls_per_step import SLS


# for implementation of a distributed model approach look at : https://www.tensorflow.org/tutorials/distribute/custom_training

class BasicExperiment():

    def __init__(self, experiment_name, model_func, opt_func, opt_params, lr_schedule_func, lr_schedule_args,
                 dataset_func, dataset_params, writer, log_path):
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

        self.name = experiment_name
        # Data
        # train_dataloader, val_dataloader, test_dataloader
        self.train_dataloader, self.val_dataloader, self.test_dataloader, self.train_set_size, self.eval_set_size, self.test_set_size, self.num_classes = dataset_func(
            **dataset_params)

        self.batch_size = dataset_params["batch_size"]
        self.steps_per_train_epoch = int(
            np.floor(self.train_set_size / self.batch_size))  # floor since we drop the last batch
        self.steps_per_val_epoch = int(np.floor(self.eval_set_size / self.batch_size))
        self.steps_per_test_epoch = int(np.floor(self.test_set_size / self.batch_size))
        # Model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if "device" in inspect.getfullargspec(model_func).args:
            self.model = model_func(num_classes=self.num_classes, device=self.device)
        elif len(inspect.getfullargspec(model_func).args) == 0:
            self.model = model_func()
        else:
            self.model = model_func(num_classes=self.num_classes)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        print(self.model)
        self.lr_scheduler = None

        if "plot_save_dir" in inspect.getfullargspec(opt_func).args:
            opt_params["plot_save_dir"] = os.path.join(self.log_path, experiment_name, "lines/")
        if "batch_size" in inspect.getfullargspec(opt_func).args:
            opt_params["batch_size"] = self.batch_size

        if "model" in inspect.getfullargspec(opt_func).args:
            opt_params["model"] = self.model
        else:
            opt_params["params"] = self.model.parameters()

        if opt_func is SLS:
            opt_params["n_batches_per_epoch"] = self.steps_per_train_epoch

        if "writer" in inspect.getfullargspec(opt_func).args:
            opt_params["writer"] = writer

        self.optimizer = opt_func(**opt_params)

        if lr_schedule_func != None:
            self.lr_scheduler = lr_schedule_func(self.optimizer, **lr_schedule_args)
        else:
            self.lr_scheduler = None

        self.loss = torch.nn.CrossEntropyLoss()
        self.loss2 = torch.nn.CrossEntropyLoss(reduction="none")

        # Checkpoints
        self.best_checkpoint_path = os.path.join(self.log_path, experiment_name, "ckpt/best.ckpt")
        # Save Weights
        self.weight_file_created = False
        self.gradient_file_created = False
        self.save_weights_path = os.path.join(self.log_path, experiment_name, "weights/weights.hdf5")
        self.save_gradients_path = os.path.join(self.log_path, experiment_name, "weights/gradients.hdf5")
        self.num_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Logging
        self.writer = writer
        # self.writer = PickleWriter(os.path.join(self.log_path, experiment_name, "tensorboard/"),
        #                           os.path.join(self.log_path, experiment_name, "pickle", experiment_name))
        self.trained_epochs = 0
        self.trained_steps = 0
        # Properties
        self.current_learning_rate = -1
        self.best_validation_accuracy = -np.inf

        self.training_acc = 0
        self.training_loss = np.inf
        self.validation_acc = 0
        self.validation_loss = np.inf
        self.current_learning_rate = 0

    def save_checkpoint_if_best(self, validation_acc, epoch):
        if validation_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = validation_acc
            os.makedirs(os.path.dirname(self.best_checkpoint_path) + '/', exist_ok=True)
            with open(self.best_checkpoint_path, 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'validation_acc': validation_acc
                }, f)
            print("overwrote best model checkpoint with validation acc: {0:0.5f}".format(
                validation_acc, epoch))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        print("loaded model checkpoint with validation acc {0:0.5f} from epoch {1:3d}".format(
            checkpoint["validation_acc"], checkpoint["epoch"]))
        return checkpoint["validation_acc"], checkpoint["epoch"]

    def train(self, steps_to_train, eval_after_steps=None):
        if eval_after_steps is None:
            eval_after_steps = self.steps_per_train_epoch
        epoch_time_measure_start = time.time()
        training_loss = 0
        training_correct = 0
        training_total = 0
        for train_step in range(steps_to_train):
            self.model.train()
            # begin = time.time()
            self.trained_steps += 1
            train_batch_loss, outputs, learning_rate, y = self._training_step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            train_batch_loss = self._to_python(train_batch_loss)
            learning_rate = self._to_python(learning_rate)
            if learning_rate is not None and not np.isnan(learning_rate):
                self.current_learning_rate = learning_rate
            if train_batch_loss is not None and learning_rate is not None and outputs is not None:
                training_loss += train_batch_loss
                _, predicted = outputs.max(1)
                training_total += y.size(0)
                training_correct += predicted.eq(y).sum().item()
                self.writer.add_scalar('training/learning rate', self.current_learning_rate,
                                       self.trained_steps)

            if train_step == 0 and train_batch_loss is not None:
                print("Initial Training Loss: {0:2.6f}".format(train_batch_loss))
                self.writer.add_scalar('training/loss', train_batch_loss, 0)

            if ((train_step + 1) % eval_after_steps is 0 or train_step is steps_to_train - 1):  # one epoch
                self.trained_epochs += 1

                if training_total / self.batch_size > eval_after_steps / 4 and not isinstance(self.optimizer,
                                                                                              PLS):  # at least a quarter epoch was trained
                    self.training_acc = training_correct / training_total
                    self.training_loss = training_loss / (training_total / self.batch_size)
                    self.writer.add_scalar('training/accuracy', self.training_acc, self.trained_epochs)
                    self.writer.add_scalar('training/loss', self.training_loss, self.trained_epochs)

                self.validation_acc, self.validation_loss = self._evaluate()

                epoch_time_measure_end = time.time()
                epoch_time_measure_in_sec = (epoch_time_measure_end - epoch_time_measure_start)
                epoch_time_measure_start = epoch_time_measure_end

                self.writer.add_scalar('data/time', epoch_time_measure_in_sec, self.trained_epochs)

                epoch_max_memory_cached = torch.cuda.max_memory_reserved() / 2 ** 30
                epoch_max_memory_allocated = torch.cuda.max_memory_allocated() / 2 ** 30
                self.writer.add_scalar('data/max_memory_cached', epoch_max_memory_cached, self.trained_epochs)
                self.writer.add_scalar('data/max_memory_allocated', epoch_max_memory_allocated, self.trained_epochs)

                torch.cuda.reset_peak_memory_stats()

                epoch_template = (
                    "\nEpoch: {:d}, Step: {:d}  Training Loss: {:2.6f}, Training Accuracy: {:2.6f}, Validation Loss: {:2.6f}, "
                    "Validation Accuracy: {:2.6f}, Learning Rate: {:2.6f},\n\t\tTime needed: {:4.2f}sec, Max GPU Memory Allocated: {:2.4f}GB, Max GPU Memory Cached: {:2.4f}GB")
                print(epoch_template.format(self.trained_epochs, self.trained_steps, self.training_loss,
                                            self.training_acc, self.validation_loss,
                                            self.validation_acc, self.current_learning_rate,
                                            epoch_time_measure_in_sec, epoch_max_memory_allocated,
                                            epoch_max_memory_cached))
                self.save_checkpoint_if_best(self.validation_acc, self.trained_epochs)
                training_loss = 0
                training_correct = 0
                training_total = 0
            # end = time.time()
            # print("time needed for one iteration: ", end - begin)

    def _to_python(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return x

    def _training_step(self):
        # if isinstance(self.optimizer, ELF) and self.optimizer.use_validation_set:
        #     x, y = next(self.val_dataloader)
        # else:
        x, y = next(self.train_dataloader)
        if not isinstance(self.optimizer, LabPal):
            x = x.to(self.device)
            y = y.to(self.device)
        output_placeholder = []

        if isinstance(self.optimizer, LabPal):
            closure = self.optimizer.get_closure(x, y, self.model, self.loss, self.device)

            loss, step_size = self.optimizer.step(closure)

            return loss, closure.output, step_size, y.to(self.device)

        # if  isinstance(self.optimizer, LabPal):
        #     x.cpu()
        #     y.cpu()
        #     def closure():
        #         x_ = x.to(self.device)
        #         y_ = y.to(self.device)
        #         self.optimizer.zero_grad()
        #         output = self.model(x_)
        #         loss_ = self.loss(output, y_)
        #         output_placeholder.append(output)
        #         return loss_
        #
        #     loss, step_size = self.optimizer.step(closure)
        #     if len(output_placeholder) > 0:
        #         return loss, output_placeholder[0], step_size, y.to(self.device)
        #     else:
        #         return loss, None, step_size, y.to(self.device)
        if isinstance(self.optimizer, PalOptimizer) \
                or isinstance(self.optimizer, SLS) \
                or isinstance(self.optimizer, GOLSI):
            def closure():
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_ = self.loss(output, y)
                output_placeholder.append(output)
                return loss_

            loss, step_size = self.optimizer.step(closure)
            if len(output_placeholder) > 0:
                return loss, output_placeholder[0], step_size, y
            else:
                return loss, None, step_size, y
        elif isinstance(self.optimizer, PLS):
            def closure():
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_f = torch.nn.CrossEntropyLoss(reduction="none")
                losses = loss_f(output, y)
                output_placeholder.append(output)
                return losses

            loss, step_size = self.optimizer.step(closure)
            assert len(output_placeholder) >= 1
            return loss, output_placeholder[0], step_size, y

        else:
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss(outputs, y)
            loss.backward()
            self.optimizer.step()
            step_size = self.optimizer.param_groups[0]['lr'] if 'lr' in self.optimizer.param_groups[0] else None
            return loss, outputs, step_size, y

    def _evaluate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        for eval_step in range(self.steps_per_val_epoch):
            x, y = next(self.val_dataloader)
            x = x.to(self.device)  # list(image.to(self.device) for image in x)
            y = y.to(self.device)

            outputs = self.model(x)
            loss = self.loss(outputs, y)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        validation_acc = correct / total
        validation_loss = val_loss / self.steps_per_val_epoch

        validation_acc = self._to_python(validation_acc)
        validation_loss = self._to_python(validation_loss)

        self.writer.add_scalar('validation/accuracy', validation_acc, self.trained_epochs)
        self.writer.add_scalar('validation/loss', validation_loss, self.trained_epochs)

        return validation_acc, validation_loss

    def test(self):
        print("*" * 30)
        print("Start Testing")
        print("*" * 30)
        validation_acc, epoch = self.load_checkpoint(self.best_checkpoint_path)
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        epoch_time_measure_start = time.time()
        for eval_step in range(self.steps_per_test_epoch):
            x, y = next(self.test_dataloader)
            x = x.to(self.device)  # list(image.to(self.device) for image in x)
            y = y.to(self.device)

            outputs = self.model(x)
            loss = self.loss(outputs, y)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        epoch_time_measure_end = time.time()
        epoch_time_measure_in_sec = (epoch_time_measure_end - epoch_time_measure_start)

        test_acc = correct / total
        test_loss = test_loss / self.steps_per_val_epoch
        self.writer.add_scalar('test/accuracy', test_acc, 0)
        self.writer.add_scalar('test/loss', test_loss, 0)

        test_template = (
            "\n Epoch {0:d}, Test Loss: {1:2.6f}, Test Accuracy: {2:2.6f}, "
            "Validation Accuracy: {3:2.6f}, Learning Rate: {4:2.6f}, Time needed: {5:4.2f}sec")
        print(test_template.format(epoch, test_loss,
                                   test_acc,
                                   validation_acc, self.current_learning_rate,
                                   epoch_time_measure_in_sec))

        return test_loss, test_acc

    def save(self):
        self.writer.flush_to_pickle()
