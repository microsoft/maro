import math
import numpy as np
import scipy.optimize
import torch

from os import path


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def to_device(device, *args):  # may not be necessary
    return [x.to(device) for x in args]


def get_flat_params_from(model):  # get 1D params
    params = []
    for param in model.parameters():
        params.append(param.view(-1))  # params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):  # set params to model
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))  # param's var count
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size()),
        )
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):  # get 1D grad,
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(torch.zeros(param.view(-1).shape))  # TODO:If grad is None, ss it right to append zeros?
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def conjugate_gradients(Avp_f, b, cg_iters=10, callback=None, residual_tol=1e-10):
    """
    conjugate gradient calculation (Ax = b)

    :param Avp_f: (function) The function describing the Matrix A dot the vector x
                  (x being the input parameter of the function)
    :param b: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    x = torch.zeros(b.size(), device=b.device)  # vector x, where Ax = b
    r = b.clone()  # residual
    p = b.clone()  # the first basis vector.
    r_dot_r = torch.dot(r, r)  # L2 norm of the residual
    for i in range(cg_iters):
        Avp = Avp_f(p)
        alpha = r_dot_r / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_r_dot_r = torch.dot(r, r)
        betta = new_r_dot_r / r_dot_r
        p = r + betta * p
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x


def line_search(model, f, x, full_step, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    f_val = f(True).item()  # f_val = f().data
    for step_frac in [0.5**x for x in range(max_backtracks)]:  # enumerate(.5**np.arange(max_backtracks))
        x_new = x + step_frac * full_step
        set_flat_params_to(model, x_new)
        f_val_new = f(True).item()  # f_val_new = f().data returns a tensor of value, item(0) returns the value
        actual_improve = f_val - f_val_new
        expected_improve = expected_improve_full * step_frac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:  # and actual_improve.item() > 0
            return True, x_new
    return False, x


def trpo_step(policy_net, value_net, states, actions, returns, advantages, max_kl, damping, l2_reg, use_fim=True):
    # TODO: need to separate the following steps into _get_actor_loss and _get_critic_loss of TrainOps.
    """update critic"""

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # weight decay. TODO:The next two lines needs careful consideration.
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        get_flat_params_from(value_net).detach().cpu().numpy(),
        maxiter=25,
    )
    set_flat_params_to(value_net, torch.tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """define the loss function for TRPO"""

    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()

    """directly compute Hessian*vector from KL"""

    def Fvp_direct(v):
        kl = policy_net.get_kl(states)  # KL-Divergence
        kl = kl.mean()  # mean KL-Divergence

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    Fvp = Fvp_direct

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = torch.sqrt(max_kl / shs)
    full_step = stepdir * lm
    expected_improve = -loss_grad.dot(full_step)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, full_step, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), "../assets"))
