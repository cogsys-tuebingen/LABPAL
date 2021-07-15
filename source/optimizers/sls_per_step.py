import torch
import copy
import time


# This version of Sls was adapted in a way that each train_step just processes one new batch. Thus a new_direction and a backtracking mode were introduced.
# the functionaliry of the algorithm is not changed.
class SLS(torch.optim.Optimizer):
    """Implements stochastic line search
    `paper <https://arxiv.org/abs/1905.09997>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        beta_f (float, optional): factor used by Goldstein for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        eta_max (float, optional): an upper bound used by Goldstein on the step size (default: 10)
        bound_step_size (bool, optional): a flag used by Goldstein for whether to bound the step-size (default: True)
        line_search_fn (float, optional): the condition used by the line-search to find the
                    step-size (default: Armijo)
    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="armijo"):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        c=c,
                        beta_b=beta_b,
                        gamma=gamma,
                        beta_f=beta_f,
                        reset_option=reset_option,
                        eta_max=eta_max,
                        bound_step_size=bound_step_size,
                        line_search_fn=line_search_fn)
        super().__init__(params, defaults)

        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

        self.state['mode'] = "new_direction"  # or "backtracking"

    def step(self, closure):
        # deterministic closure
        if self.state['mode'] == "new_direction":
            seed = time.time()
            self.state['step'] += 1

            def closure_deterministic():
                with random_seed_torch(int(seed)):
                    return closure()

            batch_step_size = self.state['step_size']

            # get loss and compute gradients
            self.zero_grad()
            loss = closure_deterministic()
            loss.backward()

            # increment # forward-backward calls
            self.state['n_forwards'] += 1
            self.state['n_backwards'] += 1

            # loop over parameter groups

            group = self.param_groups[0]
            params = group["params"]

            # save the current parameters:
            params_current = copy.deepcopy(params)
            grad_current = get_grad_list(params)

            grad_norm = compute_grad_norm(grad_current)

            step_size = reset_step(step_size=batch_step_size,
                                   n_batches_per_epoch=group['n_batches_per_epoch'],
                                   gamma=group['gamma'],
                                   reset_option=group['reset_option'],
                                   init_step_size=group['init_step_size'])

            self.state['grad_norm'] = grad_norm
            self.state['step_size'] = step_size
            if grad_norm >= 1e-8:
                self.state['mode'] = "backtracking"
                self.state['num_backtracking_steps'] = 0
                self.state['grad_current'] = copy.deepcopy(grad_current)
                self.state['params_current'] = params_current
                self.state['first_loss'] = loss
                self.state['step_size_new'] = step_size
                self.closure = closure_deterministic
            # rint(step_size)
            return loss, self.state['step_size']

            # only do the check if the gradient norm is big enough
        else:
            with torch.no_grad():
                group = self.param_groups[0]
                params = group["params"]
                grad_norm = self.state['grad_norm']
                step_size_old = self.state['step_size']
                grad_current = self.state['grad_current']
                params_current = self.state['params_current']
                loss = self.state['first_loss']
                closure_deterministic = self.closure
                self.state['num_backtracking_steps'] += 1

                step_size = self.state['step_size_new']

                # try a prospective step
                try_sgd_update(params, step_size, params_current, grad_current)

                # compute the loss at the next step; no need to compute gradients.
                loss_next = closure_deterministic()
                self.state['n_forwards'] += 1

                # =================================================
                # Line search

                armijo_results = check_armijo_conditions(step_size=step_size,
                                                         step_size_old=step_size_old,
                                                         loss=loss,
                                                         grad_norm=grad_norm,
                                                         loss_next=loss_next,
                                                         c=group['c'],
                                                         beta_b=group['beta_b'])
                found, step_size, step_size_old = armijo_results
                self.state['step_size_new'] = step_size
                if found == 1:
                    self.state['mode'] = "new_direction"
                    self.state['step_size'] = step_size

                if found == 0 and self.state['num_backtracking_steps'] == 100:
                    try_sgd_update(params, 1e-6, params_current, grad_current)
                    self.state['mode'] = "new_direction"
                    self.state['step_size'] = step_size  # 1e-6 # strange but as in original code

            return None, None


## SLS utils:
import torch
import torch.cuda

import numpy as np
import contextlib


def check_armijo_conditions(step_size, step_size_old, loss, grad_norm,
                            loss_next, c, beta_b):
    found = 0

    # computing the new break condition
    break_condition = loss_next - \
                      (loss - (step_size) * c * grad_norm ** 2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old


def check_goldstein_conditions(step_size, loss, grad_norm,
                               loss_next,
                               c, beta_b, beta_f, bound_step_size, eta_max):
    found = 0
    if (loss_next <= (loss - (step_size) * c * grad_norm ** 2)):
        found = 1

    if (loss_next >= (loss - (step_size) * (1 - c) * grad_norm ** 2)):
        if found == 1:
            found = 3  # both conditions are satisfied
        else:
            found = 2  # only the curvature condition is satisfied

    if (found == 0):
        raise ValueError('Error')

    elif (found == 1):
        # step-size might be too small
        step_size = step_size * beta_f
        if bound_step_size:
            step_size = min(step_size, eta_max)

    elif (found == 2):
        # step-size might be too large
        step_size = max(step_size * beta_b, 1e-8)

    return {"found": found, "step_size": step_size}


def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma ** (1. / n_batches_per_epoch)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current


def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]


@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state, device)
