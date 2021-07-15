# @Author: Aaron Mishkin <aaronmishkin>
# @Email:  amishkin@cs.ubc.ca

# References:

# [1] M. Mahsereci and P. Hennig. Probabilistic line searches for stochastic
# optimization. In Advances in Neural Information Processing Systems 28, pages
# 181-189, 2015.

import copy
import functools

import torch

from source.optimizers.pls_utils.prob_ls import ProbLSOptimizer
import source.optimizers.pls_utils.autograd_hacks  as autograd_hacks


class PLS(torch.optim.Optimizer):
	''' PyTorch optimizer class implementing Probabilistic Line Search as proposed in [1].
		This class wraps around the line search code written by the authors of [1]
		in the same style as their TensorFlow implementation (see https://github.com/ProbabilisticNumerics/probabilistic_line_search)
		Parameters:
			c1: Scalar parameters for the first Wolfe conditions. Default to 0.05.
			cW: Acceptance threshold for the Wolfe probability. Defaults to 0.3.
			fpush: Push factor that is multiplied with the accepted step size to get
				the base step size for the next line search. Defaults to 1.0.
			alpha0: Initial step size. Defaults to 0.01.
			target_df: The target value for the relative projected gradient
				df(t)/abs(df(0)). Defaults to 0.5.
			df_lo, df_hi: Lower and higher threshold for the relative projected
				gradient df(t)/abs(df(0)). Default to -0.1 and 1.1.
			max_steps: Maximum number of steps (function evaluations) per line
				search. Defaults to 10.
			max_epl: Maximum number of exploration steps per line search. Defaults
			to 6.
			max_dmu0: If the posterior derivative at t=0 exceeds ``max_dmu0``, the
				current line search is aborted as a safeguard against bad search
				directions. Defaults to 0.0.
			max_change_factor: The algorithm usually takes the accepted alpha of the
				current line search as the base ``alpha0`` of the next one (after
				multiplying with ``fpush``). However, if a line search accepts an
				alpha that is more than ``max_change_factor`` times smaller or larger
				than the current ``alpha0``, we instead set the next ``alpha0`` to a
				running average of the accepted alphas (``alpha_stats``). Defaults to
				10.0.
			expl_policy: String indicating the policy used for exploring points *to
				the right* in the line search. If ``k`` is the number of exploration
				steps already made, then the ``"linear"`` exploration policy chooses
				``2*(k+1)*alpha0`` as the next exploration candidate. The
				``"exponential"`` policy chooses ``2**(k+1)*alpha0``. Defaults to
				``"linear"``.
	'''

	def __init__(self, model, momentum=0.0, c1=0.05, cW=0.3, fpush=1.0, alpha0=0.01,
				 target_df=0.5, df_lo=-0.1, df_hi=1.1, max_steps=20, max_expl=20,
				 max_dmu0=0.0, max_change_factor=10.0, expl_policy="linear", plot_save_dir="Unused"):

		# set defaults:
		defaults = dict(momentum=momentum)

		# create underlying probabilistic line search object
		prob_ls = ProbLSOptimizer(c1=c1,
								  cW=cW,
								  fpush=fpush,
								  alpha0=alpha0,
								  target_df=target_df,
								  df_lo=df_lo,
								  df_hi=df_hi,
								  max_steps=max_steps,
								  max_expl=max_expl,
								  max_dmu0=max_dmu0,
								  max_change_factor=max_change_factor,
								  expl_policy=expl_policy)

		super(PLS, self).__init__(model.parameters(), defaults)

		self.prob_ls = prob_ls
		self.model = model
		autograd_hacks.add_hooks(self.model)

		self.state['step'] = 0

		self.state['dt'] = 0

		# Initialize optimizer memory with zeros:
		self.state['update_dirs'] = self._init_memory_dict()
		self.state['mem_grads'] = self._init_memory_dict()
		self.state['mem_grad_vars'] = self._init_memory_dict()

		self.state['mem_f'] = 0.
		self.state['mem_fvar'] = 0.

		self.state['forward_calls'] = 0
		self.state['backward_calls'] = 0

		# initialize complete as true for the purpose of metric computation.
		self.state['complete'] = True

	def _update_func_evals_counters(self, backward_called=False):
		# record func evals
		self.state['forward_calls'] = self.state['forward_calls'] + 1
		if backward_called:
			self.state['backward_calls'] = self.state['backward_calls'] + 1

	def step(self, closure):

		# handle initialization before first step.
		if self.state['step'] == 0:
			self._try_update(self.state['dt'], closure)
			self._accept_update()
			self._try_update(self.state['dt'], closure)
			self.state['prepared'] = True
			loss_first, df_new, mem_fvar, dfvar_new = self._accept_update()

			self.prob_ls.prepare(loss_first, df_new, mem_fvar, dfvar_new)

		accept_func = self._accept_update
		adv_eval_func = functools.partial(self._try_update, closure=closure)

		loss, complete, step_size = self.prob_ls.proceed(accept_func, adv_eval_func)

		self.state['complete'] = complete

		self.state['step'] = self.state['step'] + 1
		if self.state['step'] == 1:
			return loss_first, step_size
		else:
			return loss, step_size

	def _accept_update(self):
		self._compute_new_dirs()
		with torch.no_grad():
			df_new = 0
			dfvar_new = 0
			for i, group in enumerate(self.param_groups):
				proj_grad_group = []
				for j, p in enumerate(group['params']):
					# Compute projected gradient df (w.r.t. the current search direction)
					dot_prod = torch.sum(torch.mul(self.state['mem_grads'][i][j], self.state['update_dirs'][i][j]))
					df_new = df_new + dot_prod

					# compute gradient variance
					dfvar_new = dfvar_new + torch.sum(
						torch.mul(self.state['mem_grad_vars'][i][j], self.state['update_dirs'][i][j] ** 2))

		return self.state['mem_f'].detach().cpu().numpy(), df_new.detach().cpu().numpy(), self.state[
			'mem_fvar'].detach().cpu().numpy(), dfvar_new.detach().cpu().numpy()

	def _try_update(self, dt, closure):
		'''
		Try a parameter update to x_current with the given update vector.
		'''
		# save the current parameters:
		x_current = copy.deepcopy(self.param_groups)

		with torch.no_grad():
			for i, group in enumerate(self.param_groups):
				for j, p in enumerate(group['params']):
					# update models parameters using SGD update
					p.data = p.data + dt * self.state['update_dirs'][i][j]
		# call the closure to get loss and compute gradients
		autograd_hacks.clear_backprops(self.model)
		self.zero_grad()
		losses = closure()
		loss = torch.mean(losses)
		loss.backward()
		autograd_hacks.compute_grad_mom(self.model)

		self._update_func_evals_counters(backward_called=True)

		# save the gradient at the current parameters:
		gradient, gradient_moments, grad_norm = get_grads(list(self.model.parameters()))

		with torch.no_grad():
			# loss = torch.mean(losses)
			fvar = torch.std(losses) ** 2

			df = 0
			dfvar = 0
			for i, group in enumerate(self.param_groups):
				for j, p in enumerate(group['params']):
					# Compute projected gradient df (w.r.t. the current search direction)
					dot_prod = torch.sum(torch.mul(gradient[i][j], self.state['update_dirs'][i][j]))
					df = df + dot_prod

					# compute gradient variance
					grad_var = (gradient_moments[i][j] - gradient[i][j] ** 2) / (losses.numel() - 1)
					dfvar = dfvar + torch.sum(torch.mul(grad_var, self.state['update_dirs'][i][j] ** 2))
					self.state['mem_grad_vars'][i][j] = grad_var
					self.state['mem_grads'][i][j] = torch.zeros_like(gradient[i][j].data).copy_(gradient[i][j].data)

			# Store information in case this iterate is accepted.
			# - gradients
			# - gradient moments
			# - f and fvar

			self.state['mem_f'] = loss
			self.state['mem_fvar'] = fvar

			result = (loss.detach().cpu().numpy(), df.detach().cpu().numpy(), fvar.detach().cpu().numpy(),
					  dfvar.detach().cpu().numpy())

			return result

	def _compute_new_dirs(self):

		for i, group in enumerate(self.param_groups):

			for j, p in enumerate(group['params']):
				neg_gradient = - torch.zeros_like(self.state['mem_grads'][i][j].data).copy_(
					self.state['mem_grads'][i][j].data)
				if self.defaults['momentum'] is None:
					dir = neg_gradient
				else:
					dir = (self.defaults['momentum'] * self.state['update_dirs'][i][j]) + neg_gradient

				self.state['update_dirs'][i][j] = dir

	def _init_memory_dict(self):
		mems = []
		with torch.no_grad():
			for i, group in enumerate(self.param_groups):
				mem_group = []
				for j, p in enumerate(group['params']):
					# initialize memory with zeros as in TensorFlow implementation.
					mem_group.append(torch.zeros_like(p.data))
				mems.append(mem_group)

		return mems


# Helper function for getting gradients:
def get_grads(param_groups):
	'''
		Extracts gradients attached to current model parameters and computes
		their 2-norm (Frobenius for matrices).
	'''
	grad_norm = 0
	gradient = []
	gradient_mom = []

	if not isinstance(param_groups[0], dict):
		param_groups = [{'params': param_groups}]

	for i, group in enumerate(param_groups):
		grad_group = []
		grad_mom_group = []
		for j, p in enumerate(group['params']):
			grad_copy = torch.zeros_like(p.grad.data).copy_(p.grad.data)
			grad_group.append(grad_copy)
			grad_norm = grad_norm + torch.sum(torch.mul(grad_copy, grad_copy))

			grad_mom_copy = torch.zeros_like(p.grad.data).copy_(p.grad_mom.data)
			grad_mom_group.append(grad_mom_copy)

		gradient.append(grad_group)
		gradient_mom.append(grad_mom_group)

	return gradient, gradient_mom, torch.sqrt(grad_norm)
