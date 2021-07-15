"""
Library for extracting interesting quantites from autograd, see README.md
Not thread-safe because of module-level variables
Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)
Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias
A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

_supported_layers = ['Linear', 'Conv2d', 'BatchNorm2d']  # Supported layer class types
_hooks_disabled: bool = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False  # global switch to catch double backprop errors on Hessian computation
_check_gradients_correct = False


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.
    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.
    Call "remove_hooks(model)" to disable this.
    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            if _layer_type(layer) is "BatchNorm2d":
                handles.append(layer.register_forward_hook(_capture_outputs))
            else:
                handles.append(layer.register_forward_hook(_capture_activations))

            # handles.append(layer.register_forward_hook(_capture_outputs))
            # handles.append(layer.register_forward_hook(_capture_activations))
            print(layer)
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)



def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach().clone())  # inplace += might change the output later


def _capture_outputs(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "outputs", output.detach().clone())  # inplace += might change the output later


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer,
                           'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach().clone())  # inplace += might change the output later
    a=2


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()
    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        A = layer.activations
        n = A.shape[0]
        if loss_type == 'mean':
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]

        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B)

        elif layer_type == 'Conv2d':
            A = torch.nn.functional.unfold(A, layer.kernel_size)
            B = B.reshape(n, -1, A.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)
            shape = [n] + list(layer.weight.shape)
            setattr(layer.weight, 'grad1', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))


def compute_grad_mom(model: nn.Module) -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()
    Args:
        model:
        loss_type: only mean case considered
    """
    global _check_gradients_correct
    # begin = time.time()

    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            # print("non supportd layer: ", layer_type)
            continue
        assert hasattr(layer, 'activations') or hasattr(layer,
                                                        'outputs'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(
            layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        if layer_type == 'Linear':

            A = layer.activations
            n = A.shape[0]
            B = layer.backprops_list[0] * n
            ori_B = B
            if (len(A.shape) == 2):
                A = torch.reshape(A, [A.shape[0], -1, 1])
                B = torch.reshape(B, [A.shape[0], -1, 1])
            AT = torch.transpose(A, 1, 2)
            grad = B @ AT
            sum_weight_grad_squared = _reduce_sum(grad * grad)
            if _check_gradients_correct:
                mean_weight_grad = _reduce_sum(grad) / n
                assert (torch.all(torch.isclose(layer.weight.grad, mean_weight_grad, atol=1e-5)))
            assert layer.weight.shape == sum_weight_grad_squared.shape
            setattr(layer.weight, 'grad_mom', sum_weight_grad_squared)
            if layer.bias is not None:
                sum_bias_grad_squared = _reduce_sum(ori_B * ori_B)
                if _check_gradients_correct:
                    mean_bias_grad = _reduce_sum(ori_B) / n
                    assert (torch.all(torch.isclose(layer.bias.grad, mean_bias_grad, atol=1e-5)))
                setattr(layer.bias, 'grad_mom', sum_bias_grad_squared)

        elif layer_type == 'Conv2d':
            AI = layer.activations
            n = AI.shape[0]
            B = layer.backprops_list[0] * n
            ori_B = B
            if layer.groups is 1:
                A = torch.nn.functional.unfold(AI, layer.kernel_size, dilation=layer.dilation, padding=layer.padding,
                                               stride=layer.stride)
                # B = B.reshape(n, -1, A.shape[-1])
                # AT = torch.transpose(A, 1, 2)
                # grad = B @ AT
                # is identical to:
                # grad = torch.einsum('ijk,ilk->ijl', B, A)
                B = B.reshape(n, -1, A.shape[-1])
                AT = torch.transpose(A, 1, 2)
                grad = B @ AT

                shape = [n] + list(layer.weight.shape)
                grad = grad.reshape(shape)
                sum_weight_grad_squared = _reduce_sum(grad * grad)
                if _check_gradients_correct:
                    mean_weight_grad = _reduce_sum(grad) / n
                    assert (torch.all(torch.isclose(layer.weight.grad, mean_weight_grad, atol=1e-4)))
                assert layer.weight.shape == sum_weight_grad_squared.shape
                setattr(layer.weight, 'grad_mom', sum_weight_grad_squared)
                if layer.bias is not None:
                    Bb = torch.sum(ori_B, [2, 3])
                    sum_bias_grad_squared = _reduce_sum(Bb * Bb)
                    if _check_gradients_correct:
                        mean_bias_grad = _reduce_sum(Bb) / n
                        assert (torch.all(torch.isclose(layer.bias.grad, mean_bias_grad, atol=1e-5)))
                    setattr(layer.bias, 'grad_mom', sum_bias_grad_squared)

            else:
                num_channels_per_group = AI.shape[1] // layer.groups
                shape = [n] + list(layer.weight.shape)
                A = torch.nn.functional.unfold(AI, layer.kernel_size, dilation=layer.dilation,
                                               padding=layer.padding, stride=layer.stride)
                A = A.reshape(n, layer.groups, -1, A.shape[-1])
                AT = torch.transpose(A, 2, 3)
                B_r = ori_B.reshape(n, layer.groups, num_channels_per_group, -1)
                full_grad = B_r @ AT
                full_grad = full_grad.reshape(shape)
                # ca. 30 times slower version, but easier to understand:
                # num_channels_per_group = int(AI.shape[1] / layer.groups)
                ##grads = []
                # shape[1] = 1
                # for group in range(layer.groups):
                #     A_c = AI[:, group * num_channels_per_group:(group + 1) * num_channels_per_group, :]
                #     B_c = B[:, group * num_channels_per_group:(group + 1) * num_channels_per_group, :]
                #     A_cu = torch.nn.functional.unfold(A_c, layer.kernel_size, dilation=layer.dilation,
                #                                       padding=layer.padding, stride=layer.stride)
                #     B_c = B_c.reshape(n, -1, A_cu.shape[-1])
                #     AT = torch.transpose(A_cu, 1, 2)
                #     grad = B_c @ AT
                #     grad = grad.reshape(shape)
                #     grads.append(grad)
                # full_grad = torch.cat(grads, 1)
                sum_weight_grad_squared = _reduce_sum(full_grad * full_grad)
                assert layer.weight.shape == sum_weight_grad_squared.shape
                if _check_gradients_correct:
                    mean_weight_grad = _reduce_sum(full_grad) / n
                    assert (torch.all(torch.isclose(layer.weight.grad, mean_weight_grad, atol=1e-5)))
                setattr(layer.weight, 'grad_mom', sum_weight_grad_squared)
                if layer.bias is not None:
                    Bb = torch.sum(ori_B, [2, 3])
                    sum_bias_grad_squared = _reduce_sum(Bb * Bb)
                    if _check_gradients_correct:
                        mean_bias_grad = _reduce_sum(Bb) / n
                        assert (torch.all(torch.isclose(layer.bias.grad, mean_bias_grad, atol=1e-5)))
                    setattr(layer.bias, 'grad_mom', sum_bias_grad_squared)


        elif layer_type == 'BatchNorm2d':
            y = layer.outputs
            n = y.shape[0]
            beta = layer.bias
            gamma = layer.weight
            yT = torch.transpose(y, 1, 3)
            epsilon = 0  # 1E-7
            AT = (yT - beta) / (gamma + epsilon)
            A = torch.transpose(AT, 1, 3)
            B = layer.backprops_list[0] * n
            gamma_grad = torch.sum(A * B, dim=[2, 3])  #
            sum_weight_grad_squared = _reduce_sum(gamma_grad * gamma_grad)
            if _check_gradients_correct:
                mean_weight_grad = _reduce_sum(gamma_grad) / n
                assert (torch.all(torch.isclose(layer.weight.grad, mean_weight_grad, atol=1e-4)).item())
            assert layer.weight.shape == sum_weight_grad_squared.shape
            setattr(layer.weight, 'grad_mom', sum_weight_grad_squared)
            if layer.bias is not None:
                grad_b = torch.sum(B, dim=[2, 3])
                sum_bias_grad_squared = _reduce_sum(grad_b * grad_b)
                if _check_gradients_correct:
                    mean_bias_grad = _reduce_sum(grad_b) / n
                    assert (torch.all(torch.isclose(layer.bias.grad, mean_bias_grad, atol=1e-5)))
                setattr(layer.bias, 'grad_mom', sum_bias_grad_squared)
    # end= time.time()
    # print("time needed to calc grad_moms: " , end-begin)


def _reduce_sum(a):
    return torch.sum(a, dim=0)


def compute_hess(model: nn.Module, ) -> None:
    """Save Hessian under param.hess for each param in the model"""

    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"

        if layer_type == 'Linear':
            A = layer.activations
            B = torch.stack(layer.backprops_list)

            n = A.shape[0]
            o = B.shape[0]

            A = torch.stack([A] * o)
            Jb = torch.einsum("oni,onj->onij", B, A).reshape(n * o, -1)
            H = torch.einsum('ni,nj->ij', Jb, Jb) / n

            setattr(layer.weight, 'hess', H)

            if layer.bias is not None:
                setattr(layer.bias, 'hess', torch.einsum('oni,onj->ij', B, B) / n)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels

            A = layer.activations.detach()
            A = torch.nn.functional.unfold(A, (Kh, Kw))  # n, di * Kh * Kw, Oh * Ow
            n = A.shape[0]
            B = torch.stack([Bt.reshape(n, do, -1) for Bt in layer.backprops_list])  # o, n, do, Oh*Ow
            o = B.shape[0]

            A = torch.stack([A] * o)  # o, n, di * Kh * Kw, Oh*Ow
            Jb = torch.einsum('onij,onkj->onik', B, A)  # o, n, do, di * Kh * Kw

            Hi = torch.einsum('onij,onkl->nijkl', Jb, Jb)  # n, do, di*Kh*Kw, do, di*Kh*Kw
            Jb_bias = torch.einsum('onij->oni', B)
            Hi_bias = torch.einsum('oni,onj->nij', Jb_bias, Jb_bias)

            setattr(layer.weight, 'hess', Hi.mean(dim=0))
            if layer.bias is not None:
                setattr(layer.bias, 'hess', Hi_bias.mean(dim=0))


def backprop_hess(output: torch.Tensor, hess_type: str) -> None:
    """
    Call backprop 1 or more times to get values needed for Hessian computation.
    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy
    Returns:
    """

    assert hess_type in ('LeastSquares', 'CrossEntropy')
    global _enforce_fresh_backprop
    n, o = output.shape

    _enforce_fresh_backprop = True

    if hess_type == 'CrossEntropy':
        batch = F.softmax(output, dim=1)

        mask = torch.eye(o).expand(n, o, o)
        diag_part = batch.unsqueeze(2).expand(n, o, o) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, o, o)

        for i in range(n):
            hess[i, :, :] = symsqrt(hess[i, :, :])
        hess = hess.transpose(0, 1)

    elif hess_type == 'LeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    for o in range(o):
        output.backward(hess[o], retain_graph=True)


def symsqrt(a, cond=None, return_rank=False, dtype=torch.float32):
    """Symmetric square root of a positive semi-definite matrix.
    See https://github.com/pytorch/pytorch/issues/25481"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B

# autograd_hacks.add_hooks(model)
# output = model(data)
# loss_fn(output, targets).backward()
# autograd_hacks.compute_grad1()
