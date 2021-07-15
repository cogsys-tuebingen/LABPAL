__author__ = ",  "
__version__ = "1.1"
__email__ = " "

import copy

import torch
from torch.optim.optimizer import Optimizer


class GOLSI(Optimizer):
    def __init__(self, params, initial_step_size=10 ** -4,
                 # # paper has 10**-8 but than alg does not work since, it is nor greater than the minimal step size
                 minimal_step_size=10 ** -8,
                 eta=2,
                 c2=0.9,
                 momentum=0.9):  #
        """Implements gradient only line search
        `paper <https://arxiv.org/pdf/1903.09383.pdf`_.
        Arguments:
            initial_step_size :
            minimal_step_size :
            eta : step increasing or decrasing factor
            c2: Wolfe condition parameter for acceptance
        """
        defaults = dict()
        super().__init__(params, defaults)

        self.current_step_size = initial_step_size  #
        self.minimal_step_size = minimal_step_size  #
        self.eta = eta
        self.c2 = c2
        self.flag = 0
        self.last_directional_derivative = None
        self.maximal_step_size = None
        self.is_first_step = True
        self.step_count = 0
        self.momentum = momentum
        self.epsilon = 1E-8
        self.direction_norm = None
        self.params_copy = None
        self.directional_derivative_at0 = 0

    def _set_direction_and_direction_norm(self):
        with torch.no_grad():
            norm = torch.tensor(0.0)
            group = self.param_groups[0]
            params = group["params"]
            for p in params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'search_direction' not in param_state:
                    buf = param_state['search_direction'] = torch.zeros_like(p.grad.data, device=p.device)
                else:
                    buf = param_state['search_direction']
                buf = buf.mul_(self.momentum)
                buf = buf.add_(-p.grad.data)
                flat_buf = buf.view(-1)
                norm = norm + torch.dot(flat_buf, flat_buf)
            norm = torch.sqrt(norm)
            if norm == 0:   norm = torch.tensor(self.epsilon)
            self.direction_norm = norm

    def _get_dierctional_derivative(self):
        with torch.no_grad():
            directional_derivative = torch.tensor(0.0)
            group = self.param_groups[0]
            params = group["params"]
            for p in params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                buf = param_state['search_direction']
                flat_buf = buf.view(-1)
                flat_grad = p.grad.data.view(-1)
                directional_derivative = directional_derivative + torch.dot(flat_grad, flat_buf)
            directional_derivative = - directional_derivative / self.direction_norm
            return directional_derivative

    def _save_weights(self):
        with torch.no_grad():
            group = self.param_groups[0]
            params = group["params"]
            params_copy = copy.deepcopy(params)
            self.params_copy = params_copy

    def _perform_step_on_line(self, step_size):
        with torch.no_grad():
            if step_size != 0:
                group = self.param_groups[0]
                params = group["params"]
                for p, p_initial in zip(params, self.params_copy):
                    param_state = self.state[p]
                    if not 'search_direction' in param_state:
                        continue
                    direction = param_state['search_direction']
                    p.data = p_initial.add(step_size * direction)

    def step(self, closure):
        """
        >>> def loss_fn():
        >>>     logits = model(inputs)
        >>>     loss_ = criterion(logits, targets)
        >>> return loss_
        """

        self.step_count += 1
        if self.flag == 0:  # new direction
            if self.is_first_step:
                #     # save current gradient as new direction in addition
                #
                #     self.is_first_step = False
                # copy gradient vars to direction vars, save new point0 weights for the line search
                self.zero_grad()
                loss = closure()
                loss.backward()
                self._set_direction_and_direction_norm()
                self._save_weights()
                self.directional_derivative_at0 = self._get_dierctional_derivative()

                self.maximal_step_size = min(1 / self.direction_norm, 10 ** 7)
                if self.current_step_size > self.maximal_step_size:
                    self.current_step_size = self.maximal_step_size
                if self.current_step_size < self.minimal_step_size:
                    self.current_step_size = self.minimal_step_size
                self.is_first_step = False
                return loss, self.current_step_size
            else:
                self._perform_step_on_line(self.current_step_size)
                self.zero_grad()
                l = closure()
                l.backward()
                directional_derivative_at_current_position = self._get_dierctional_derivative()

                tol_dd = abs(self.c2 * self.directional_derivative_at0)
                if directional_derivative_at_current_position > 0 and self.current_step_size < self.maximal_step_size:
                    self.flag = 1  # decrease step size
                if directional_derivative_at_current_position < 0 and self.current_step_size > self.minimal_step_size:
                    self.flag = 2  # increase step size
                if directional_derivative_at_current_position > 0 and directional_derivative_at_current_position < tol_dd:
                    self.flag = 0  # accept condition
                # other conditions are not handeled in the paper.
                # if self.current_step_size  < self.maximal_step_size  and  self.current_step_size  > self.minimal_step_size the line search should end
                self.is_first_step = True
                return None, None

        # in contrast to the paper we handle each input loading as one step, to make steps more comparable
        elif self.flag == 2:
            self.current_step_size = self.current_step_size * self.eta
            self.zero_grad()
            l = closure()
            l.backward()
            directional_derivative_at_current_position = self._get_dierctional_derivative()  #
            if directional_derivative_at_current_position >= 0:
                self.flag = 0
            if self.current_step_size > self.maximal_step_size / self.eta:
                self.flag = 0
            return None, None
        elif self.flag == 1:
            self.current_step_size = self.current_step_size / self.eta
            self.zero_grad()
            l = closure()
            l.backward()
            directional_derivative_at_current_position = self._get_dierctional_derivative()
            if directional_derivative_at_current_position < 0:
                self.flag = 0
            if self.current_step_size < self.minimal_step_size * self.eta:
                self.flag = 0
            return None, None
