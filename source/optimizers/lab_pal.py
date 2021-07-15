__author__ = ",  "
__version__ = "1.1"
__email__ = " "

import contextlib
import time
from enum import Enum
import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


# noinspection PyDefaultArgument
class LabPal(Optimizer):
    # noise_factor =  128/batch_size * dataset_size/40000


    def __init__(self, params=required, mode="NSGD", noise_factor=1, update_step_adaptation=1.8, momentum=0.0,
                 amount_steps_to_reuse_lr=1000, amount_batches_for_full_batch_loss_approximation=10,
                 batch_size_schedule=((0.0, 1), (75000.0, 2), (112500.0, 4)), amount_untrustworthy_initial_steps=400,
                 amount_of_former_lr_values_to_consider=25, parabolic_approximation_sample_step_size=0.01,
                 max_step_size=1.0, epsilon=1e-6, writer=None, is_print=True, logging=False):
        """
        The Large-Batch PAL optimizer.
        Approximates the full-batch loss in negative gradient direction with a one-dimensional parabolic function.
        This approximation is done rarely and measured with a large batch size.
        The learning rate is derived form the position of the minimum of the approximation and reused for
         multiple steps.
        Either SGD or normalized SGD (NSGD) using the unit gradient can be used. For SGD a learning rate is measured.
        For NSGD the step size is measured. In the following we just use the term learning rate.
        An example of how to simply implement this optimizer is given in the docu of the step function.

        @param params: net.parameters()
        @param mode: either "SGD" or "NSGD" (=normalized SGD using the unit gradient).
        @param noise_factor: factor applied to the batch sie in the learning rate schedule. If the gradient noise is higher as in the original problem of learning
        a ResNet on Cifar10 with batch size 128. E.g. mult=10 if training with a batch size of 12 on Cifar10 or
        mult=236 if training on Imagenet with batch size 128 (since the dataset is roughly 236 times larger.)
        It is given by: noise_factor =  128/batch_size * dataset_size/40000
        @param update_step_adaptation: the measured step size is multiplied by this factor.
        @param momentum: SGD alike momentum factor.
        @param amount_steps_to_reuse_lr: amount of SGD or NSGD weight updates to perform before remeasuring
        a new learning rate.
        @param amount_batches_for_full_batch_loss_approximation: the algorithm approximates the full-batch loss with
         this factor * batch size samples.
        @param batch_size_schedule: schedule which defines at which batch size to use after the a specific step.
         Tuple: (step, batch_size_to_use). Those batch sizes are measured iteratively and thus are independent
         of the maximum batch size the gpu memory supports.
        @param amount_untrustworthy_initial_steps: for the first update steps usually the parabolic property does
        not hold well, since for this amount of steps measured learning rates are not saved for later averaging.
        @param amount_of_former_lr_values_to_consider:  amount of former measurements of learning rates to average
         with the recent learning rate measurement. Reduces the error of the measurements.
        @param parabolic_approximation_sample_step_size: distance between the 3 points measured on the
        full-batch loss approximation to perform a parabolic approximation
        @param max_step_size: maximal step size tp prevent the algorithm from failure
        if a step size measure is not exact enough.
        @param epsilon: epsilon to avoid numerical instability.
        @param writer: optional tensorboard logger for detailed logs
        @param is_print: print optimizer information
        @param logging: loggs more data but makes the optimizer significantly slower. E.g. for SGD mode also the norm is computed.
        """

        if parabolic_approximation_sample_step_size <= 0.0:
            raise ValueError("Invalid measuring step size: {}".format(parabolic_approximation_sample_step_size))
        if max_step_size < 0.0:
            raise ValueError("Invalid maximal step size: {}".format(max_step_size))
        if momentum < 0.0:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if update_step_adaptation <= 0.0:
            raise ValueError("Invalid update_step_adaptation: {}".format(update_step_adaptation))
        if amount_untrustworthy_initial_steps < 0.0:
            raise ValueError("Invalid amount_untrustworthy_initial_steps {}".format(amount_untrustworthy_initial_steps))
        if amount_untrustworthy_initial_steps < 0.0:
            raise ValueError("Invalid mean_lr_values_to_consider {}".format(amount_of_former_lr_values_to_consider))
        if max_step_size <= 0.0:
            raise ValueError("Invalid max_step_size {}".format(max_step_size))
        if amount_batches_for_full_batch_loss_approximation <= 0.0:
            raise ValueError("Invalid amount_batches_for_full_batch_loss_approximation {}".format(
                amount_batches_for_full_batch_loss_approximation))
        if not ("SGD" == mode or "NSGD" == mode):
            raise ValueError("Invalid amount_batches_for_full_batch_loss_approximation {}".format(
                amount_batches_for_full_batch_loss_approximation))

        if not isinstance(parabolic_approximation_sample_step_size, torch.Tensor):
            parabolic_approximation_sample_step_size = torch.tensor(parabolic_approximation_sample_step_size)
        if not isinstance(max_step_size, torch.Tensor):
            max_step_size = torch.tensor(max_step_size)
        if not isinstance(momentum, torch.Tensor):
            momentum = torch.tensor(momentum)
        if not isinstance(update_step_adaptation, torch.Tensor):
            update_step_adaptation = torch.tensor(update_step_adaptation)

        self._params = list(params)
        self._tensorboard_logger = writer
        self._amount_of_optimizer_steps = -1
        self._current_loss = torch.tensor(0.0)
        self._loss_closures = []
        self._last_update_step = None
        self._last_lr = None
        self._last_non_averaged_lrs = []
        self._last_non_averaged_update_seps = []
        self._parabolic_approximation_sample_step_size = parabolic_approximation_sample_step_size
        self._amount_batches_for_full_batch_loss_approximation = amount_batches_for_full_batch_loss_approximation
        self._full_batch_loss_approximation_batch_counter = 0
        self._amount_weight_updates_to_reuse_lr = amount_steps_to_reuse_lr
        self._starting_step_to_reuse_learning_rate = 0
        self.mode = mode  # "SGD", or "NSGD"
        self._amount_weight_updates_performed = -1
        self._amount_untrustworthy_initial_steps = int(amount_untrustworthy_initial_steps)
        self._amount_of_former_lr_values_to_consider = int(amount_of_former_lr_values_to_consider)
        self._batch_size_schedule = self._apply_factor_to_schedule(batch_size_schedule, noise_factor)
        self._training_batch_size = 1
        self._current_sampled_batch_size_for_training = 0
        self._max_step_size = max_step_size
        self._momentum = momentum
        self._update_step_adaptation = update_step_adaptation
        self._epsilon = epsilon
        self._first_direction_norm = None
        self._is_print = is_print
        self.logging = logging

        self._internal_state = self.State.TRAINING

        self.loss_to_return = None

        defaults = dict()
        super(LabPal, self).__init__(self._params, defaults)

    class State(Enum):
        TRAINING = 0,
        APPROXIMATING_FULL_BATCH_LOSS = 1

    def get_closure(self, inputs, targets, model, loss_criterion, device):
        """
        Builds a closure that can be used with the step method.
        After calling the closure, self.output and self.loss are set.
        @param inputs: model inputs
        @param targets: ground truth targets
        @param model: the network
        @param loss_criterion:
        @param device: either gpu or cpu
        @return: callable closure.
        """
        inputs.cpu()
        targets.cpu()

        class Closure:
            self.output = None
            self.loss = None

            def __init__(self, optimizer):
                self.opimizer = optimizer

            def __call__(self):
                x_ = inputs.to(device)
                y_ = targets.to(device)
                self.opimizer.zero_grad()
                output = model(x_)
                loss_ = loss_criterion(output, y_)
                self.output = output
                self.loss = loss_
                return loss_

        return Closure(self)

    @staticmethod #
    def _apply_factor_to_schedule(schedule, mult):
        schedule = np.array(schedule)
        schedule[:, 1] = [max(int(e*mult), 1) for e in schedule[:, 1]]
        schedule = schedule.astype(int)
        return list(schedule)

    @torch.no_grad()
    def _set_direction_get_norm_and_derivative(self, params, momentum, epsilon,
                                               calc_norm_and_directional_derivative=True, reuse_direction=True):

        assert self._current_sampled_batch_size_for_training == self._training_batch_size or \
               self._full_batch_loss_approximation_batch_counter == self._training_batch_size + 1
        with torch.no_grad():
            directional_derivative = torch.tensor(0.0)
            norm = torch.tensor(0.0)
            if momentum != 0:
                for p in params:
                    param_state = self.state[p]
                    if "batch_buffer" not in param_state and self._training_batch_size != 1:
                        continue
                    if self._training_batch_size != 1:
                        gradient = param_state["batch_buffer"]
                    else:
                        gradient = p.grad
                    if 'direction_buffer' not in param_state:
                        buf = param_state['direction_buffer'] = torch.zeros_like(gradient, device=p.device)
                    else:
                        buf = param_state['direction_buffer']
                    buf = buf.mul_(momentum)
                    buf = buf.add_(gradient)
                    if calc_norm_and_directional_derivative:
                        flat_buf = buf.view(-1)
                        flat_grad = gradient.view(-1)
                        directional_derivative = directional_derivative + torch.dot(flat_grad, flat_buf)
                        norm = norm + torch.dot(flat_buf, flat_buf)
                if calc_norm_and_directional_derivative:
                    norm = torch.sqrt(norm)
                    if norm == 0:
                        norm = torch.tensor(epsilon)
                    directional_derivative = -directional_derivative / norm
            elif reuse_direction or calc_norm_and_directional_derivative or self._training_batch_size != 1:
                for p in params:
                    param_state = self.state[p]
                    if "batch_buffer" not in param_state and self._training_batch_size != 1:
                        continue
                    if self._training_batch_size != 1:
                        if reuse_direction:
                            gradient = param_state["direction_buffer"] = param_state["batch_buffer"].clone()
                        else:
                            gradient = param_state["direction_buffer"] = param_state["batch_buffer"]
                    else:
                        if reuse_direction:
                            gradient = param_state["direction_buffer"] = p.grad.clone()
                        else:
                            gradient = p.grad
                    if calc_norm_and_directional_derivative:
                        flat_grad = gradient.view(-1)
                        norm = norm + torch.dot(flat_grad, flat_grad)
                if calc_norm_and_directional_derivative:
                    norm = torch.sqrt(norm)
                    if norm == 0:
                        norm = torch.tensor(epsilon)
                    directional_derivative = -norm

        return norm, directional_derivative

    @torch.no_grad()
    def _update_direction_mean_with_new_batch(self, params, loss_fn_deterministic):
        self._current_sampled_batch_size_for_training += 1
        with torch.enable_grad():
            loss_0 = loss_fn_deterministic()
            loss_0.backward()
        self._update_current_loss(loss_0, self._current_sampled_batch_size_for_training)
        if self._training_batch_size > 1 :
            for p in params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'batch_buffer' not in param_state:
                    param_state['batch_buffer'] = torch.zeros_like(p.grad, device=p.device)
                param_state["batch_buffer"] = param_state["batch_buffer"] * (
                        self._current_sampled_batch_size_for_training - 1) / \
                                              self._current_sampled_batch_size_for_training + 1 / \
                                              self._current_sampled_batch_size_for_training * p.grad

    @torch.no_grad()
    def _perform_param_update_step(self, params, step, direction_norm, reuse_direction=False):
        with torch.no_grad():
            if step != 0:
                for p in params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if self._training_batch_size > 1 or reuse_direction:
                        neg_line_direction = param_state['direction_buffer']
                    else:
                        neg_line_direction = p.grad
                    if direction_norm != 1.0:
                        p.data.add_(step * -neg_line_direction / direction_norm)
                    else:
                        p.data.add_(step * -neg_line_direction)

    def _update_current_loss(self, loss, counter):
        self.loss_to_return = loss
        loss = self._current_loss * (counter - 1) / counter + 1 / counter * loss
        if not torch.isnan(loss):
            self._current_loss = loss

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a LabPal optimization step.
        Calls the loss_fn multiple times.
        Only one group is supported!

        :param closure: function that returns the current batch-loss.
        Your code should have the following form::
            inputs, targets = next(dataset)
            #to avoid memory issues inputs and targets have to be located on the cpu ram
            inputs.cpu()
            targets.cpu()
            loss_placeholder = []
            def closure():
                 # do not do a backward pass here!!!
                 inputs_ = inputs.to(self.device)
                 targets_ = targets.to(self.device)
                 out_ = net(inputs_)
                 loss_ = criterion(out_, targets_)
                 loss_placeholder.append(loss_)
            return loss_

           labpal.step(closure)
           loss=loss_placeholder[0]

            if len(output_placeholder) > 0:
                return loss, output_placeholder[0]

        :return: the current measurement of the batch-loss for the current parameters
        """
        self._amount_of_optimizer_steps += 1

        if len(self._batch_size_schedule) > 0 and self._amount_of_optimizer_steps == self._batch_size_schedule[0][0] \
                and self._current_sampled_batch_size_for_training == 0:
            self._training_batch_size = self._batch_size_schedule[0][1]
            self._batch_size_schedule.pop(0)
            self._internal_state = self.State.APPROXIMATING_FULL_BATCH_LOSS
            self.reset_values()
            self.print_("new training batch size is", self._training_batch_size)
            self.print_("Approximating the current full-batch loss with {} batches. Steps trained: {} ".format(
                self._amount_batches_for_full_batch_loss_approximation * self._training_batch_size,
                self._amount_of_optimizer_steps))

        seed = time.time()
        self.zero_grad()

        def loss_fn_deterministic():
            with self.random_seed_torch(int(seed)):
                closure_output = closure()
            # torch.cuda.ipc_collect() # cleans gpu memory
            return closure_output

        if self._internal_state is self.State.TRAINING and \
                (((self._amount_of_optimizer_steps - self._starting_step_to_reuse_learning_rate) //
                  self._training_batch_size) >= self._amount_weight_updates_to_reuse_lr or
                 self._amount_of_optimizer_steps == 0 or
                 self._amount_weight_updates_performed == self._amount_untrustworthy_initial_steps + 1):
            self._internal_state = self.State.APPROXIMATING_FULL_BATCH_LOSS
            self.reset_values()
            self.print_("Approximating the current full-batch loss with {} batches. Steps trained: {} ".format(
                self._amount_batches_for_full_batch_loss_approximation
                * self._training_batch_size, self._amount_of_optimizer_steps))

        if self._internal_state is self.State.APPROXIMATING_FULL_BATCH_LOSS:
            self._full_batch_loss_approximation_batch_counter += 1
            self._loss_closures.append(loss_fn_deterministic)
            if self._full_batch_loss_approximation_batch_counter <= self._training_batch_size:
                self._update_direction_mean_with_new_batch(self._params,
                                                           loss_fn_deterministic)  # updates _current_loss internally
                if self._full_batch_loss_approximation_batch_counter == self._training_batch_size:
                    first_direction_norm, directional_derivative = self._set_direction_get_norm_and_derivative(
                        self._params,
                        self._momentum,
                        self._epsilon)
                    if directional_derivative >= 0  or torch.isnan(first_direction_norm) or torch.isinf(first_direction_norm) or first_direction_norm < 10E-4 or first_direction_norm>100:
                        self.print_("-" * 100)
                        self.print_("parabolic approximation failed. first directional derivative > 0 or norm  nan or inf. Last learning rate will be used")
                        self.print_(
                            "-" * 100)
                        self._internal_state = self.State.TRAINING
                        self.reset_values()
                    else:
                        self._first_direction_norm =   max(first_direction_norm,self._epsilon)


            else:
                loss = loss_fn_deterministic()
                self._update_current_loss(loss, self._full_batch_loss_approximation_batch_counter)

            if self._full_batch_loss_approximation_batch_counter == int(
                    self._amount_batches_for_full_batch_loss_approximation * self._training_batch_size):

                mean_loss_0 = self._current_loss.item()
                measuring_step_directional_derivative = 0.000177
                # double machine precision is 10^-15 for second derivative we need 10^-15^1/4 = 0.00018
                # but if we take h to small the curvature might become locally negative
                mu = self._parabolic_approximation_sample_step_size
                self._perform_param_update_step(self._params, measuring_step_directional_derivative,
                                                self._first_direction_norm, reuse_direction=True)
                losses_mu_1 = [f().item() for f in self._loss_closures]
                mean_loss_mu_1 = np.nanmean(losses_mu_1)

                self._perform_param_update_step(self._params, -measuring_step_directional_derivative + mu,
                                                self._first_direction_norm, reuse_direction=True)
                losses_mu_2 = [f().item() for f in self._loss_closures]
                mean_loss_mu_2 = np.nanmean(losses_mu_2)


                losses = [mean_loss_0, mean_loss_mu_1, mean_loss_mu_2]
                positions = [0, measuring_step_directional_derivative, mu.item()]
                line_derivatives = np.gradient(losses, positions)
                line_curvatures = np.gradient(line_derivatives, positions)

                # parabolic parameters
                b = line_derivatives[0]
                a = line_curvatures[0] / 2

                if a > 0 > b:
                    lab_pal_step = -b / (2 * a)

                    if lab_pal_step > self._max_step_size:
                        lab_pal_step = self._max_step_size

                    lab_pal_step *= self._update_step_adaptation
                    lab_pal_step = max(lab_pal_step.item(), self._epsilon)
                    if not np.isnan(lab_pal_step) and not np.isinf(lab_pal_step) and  lab_pal_step!=None:

                        if self._amount_weight_updates_performed > self._amount_untrustworthy_initial_steps:
                            self._last_non_averaged_update_seps.append(lab_pal_step)
                            self._last_update_step = np.nanmean(
                                self._last_non_averaged_update_seps[-self._amount_of_former_lr_values_to_consider:])
                        else:
                            self._last_update_step = lab_pal_step

                        pure_lr = max(lab_pal_step / max(self._first_direction_norm, self._epsilon), self._epsilon)

                        if isinstance(pure_lr, torch.Tensor):
                            pure_lr = pure_lr.item()

                        if self._amount_weight_updates_performed > self._amount_untrustworthy_initial_steps:
                            self._last_non_averaged_lrs.append(pure_lr)

                            self._last_lr = np.nanmean(
                                self._last_non_averaged_lrs[-self._amount_of_former_lr_values_to_consider:])
                        else:
                            self._last_lr = pure_lr

                        self.print_("-" * 100)
                        self.print_("step to estimated min location: ",
                                    mu.item())
                        # we scale with the first direction norm since we do not know the real grad projected on
                        # the search direction
                        self.print_("new update step: ",
                                    self._last_update_step)
                        self.print_("new learning rate: ", self._last_lr)
                        self.print_(
                            "-" * 100)
                    else:
                        self.print_("-" * 100)
                        self.print_("parabolic approximation failed. Last learning rate will be used")
                        self.print_(
                            "-" * 100)
                else:
                    self.print_("-" * 100)
                    self.print_("parabolic approximation failed. Last learning rate will be used")
                    self.print_(
                        "-" * 100)

                if self._last_update_step is not None:
                    self._perform_param_update_step(self._params, -mu + self._last_update_step,
                                                    self._first_direction_norm,reuse_direction=True)
                    self._starting_step_to_reuse_learning_rate = self._amount_of_optimizer_steps
                    self._internal_state = self.State.TRAINING
                    self._amount_weight_updates_performed += 1
                    self.reset_values()
                else:
                    self.print_("Any learning rate has not been estimated, yet. Performing measurement again.")
                    self._perform_param_update_step(self._params, -mu,
                                                    self._first_direction_norm,reuse_direction=True)
                    self.reset_values()

        elif self._internal_state is self.State.TRAINING and self.mode == "NSGD":
            # does self._current_sampled_batch_size_for_training+=1
            self._update_direction_mean_with_new_batch(self._params,
                                                       loss_fn_deterministic)
            if self._current_sampled_batch_size_for_training == self._training_batch_size:

                loss_0 = self._current_loss
                direction_norm, _ = self._set_direction_get_norm_and_derivative(self._params, self._momentum,
                                                                                self._epsilon,reuse_direction = False)
                self._perform_param_update_step(self._params, self._last_update_step, direction_norm)
                self._amount_weight_updates_performed += 1

                self._last_lr = max(self._last_update_step / max(direction_norm, self._epsilon), self._epsilon)
                self._current_sampled_batch_size_for_training = 0

                self._current_loss = torch.tensor(0.0)

                if self._tensorboard_logger is not None and self.logging:
                    s = "batch"
                    t = self._amount_of_optimizer_steps
                    self._tensorboard_logger.add_scalar('train-%s/l_0' % s, loss_0.item(), t)
                    self._tensorboard_logger.add_scalar('train-%s/s_upd' % s, self._last_update_step, t)
                    self._tensorboard_logger.add_scalar('train-%s/lr' % s, self._last_update_step / direction_norm,
                                                        t)
                    self._tensorboard_logger.add_scalar('train-%s/direction_norm' % s, direction_norm.item(), t)
                    self._tensorboard_logger.add_scalar('train-%s/training_batch_size' % s,
                                                        self._training_batch_size, t)

        elif self._internal_state is self.State.TRAINING and self.mode == "SGD":
            # does self._current_sampled_batch_size_for_training+=1
            self._update_direction_mean_with_new_batch(self._params,
                                                       loss_fn_deterministic)
            if self._training_batch_size == self._current_sampled_batch_size_for_training:
                self._amount_weight_updates_performed += 1
                loss_0 = self._current_loss

                if self.logging:
                    norm, _ = self._set_direction_get_norm_and_derivative(self._params, self._momentum,
                                                                          self._epsilon, calc_norm_and_directional_derivative = True, reuse_direction = False)
                else:
                    self._set_direction_get_norm_and_derivative(self._params, self._momentum,
                                                                          self._epsilon, calc_norm_and_directional_derivative = False, reuse_direction = False)

                self._perform_param_update_step(self._params, self._last_lr, 1.0)
                self._current_sampled_batch_size_for_training = 0
                self._current_loss = torch.tensor(0.0)
                if self._tensorboard_logger is not None and self.logging:
                    # cur_time = int((time.time() - self.time_start) * 1000)
                    # log in ms since it has to be an integer
                    s = "batch"
                    t = self._amount_of_optimizer_steps
                    self._tensorboard_logger.add_scalar('train-%s/l_0' % s, loss_0.item(), t)
                    # self._tensorboard_logger.add_scalar('train-%s/mss' % s, self._max_step_size, t)
                    self._tensorboard_logger.add_scalar('train-%s/s_upd' % s, self._last_lr * norm, t)
                    self._tensorboard_logger.add_scalar('train-%s/lr' % s, self._last_lr, t)
                    self._tensorboard_logger.add_scalar('train-%s/norm' % s, norm, t)
                    self._tensorboard_logger.add_scalar('train-%s/training_batch_size' % s,
                                                        self._training_batch_size, t)
        else:
            raise Exception("invalid mode. Use SGD, NSGD")
        return self.loss_to_return, self._last_lr

    def print_(self, *args):
        if self._is_print:
            print(*args)

    @torch.no_grad()
    def reset_values(self):
        self._full_batch_loss_approximation_batch_counter = 0
        self._current_sampled_batch_size_for_training = 0
        self._current_loss = torch.tensor(0.0)
        self._loss_closures = []

    @contextlib.contextmanager
    def random_seed_torch(self, seed):
        """
    source: https://github.com/IssamLaradji/sls/
    """
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            # noinspection PyTypeChecker
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)


