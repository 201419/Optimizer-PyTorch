import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


def scaled_sign(x):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm and divided by the number of elements
    """
    return x.norm(p=1) / x.nelement() * torch.sign(x)


def unscaled_sign(x):
    """
    This is the standard sign compression. It has been experimented to give worse test accuracies than the scaled
    counter part.
    :param x: torch Tensor
    :return: sign(tensor)
    """
    return torch.sign(x)


class ErrorFeedbackSGD(Optimizer):
    r"""Implements the error feedback stochastic gradient descent with memory (optionally with momentum).
        It handles parameters groups separately. The implementation largely follows the one of SGD.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        comp (function): The compression operator to be applied
        memory (bool, False by default)

    Example:
        >>> optimizer = ErrorFeedbackSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, comp='scaled_sign', memory=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if comp == 'scaled_sign':
            comp = scaled_sign
        elif comp == 'sign':
            comp = unscaled_sign
        elif not callable(comp) and comp is not None:
            raise ValueError("Invalid comp value: {} (must be callable or None)".format(comp))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        comp=comp, memory=memory)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ErrorFeedbackSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)

                # To compute the gradients norms ratios over time
                param_state['dim'] = p.nelement()
                param_state['gradient'] = None
                param_state['corrected_gradient'] = None

    def __setstate__(self, state):
        super(ErrorFeedbackSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            comp = group['comp']
            memory = group['memory']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # d_p corresponds to g in alg. 1 from the paper.
                param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later

                d_p = group['lr'] * d_p
                corrected_gradient = param_state['memory'] + d_p

                # Save the corrected gradient to compute the norms
                param_state['corrected_gradient'] = corrected_gradient

                if comp is not None:
                    corrected_gradient = comp(corrected_gradient)

                ''' hack to scale the signed gradient by the learning
                    rate since torch.sign(x) ignores the learning rate '''
                if comp == unscaled_sign:
                    corrected_gradient = group['lr'] * corrected_gradient

                if memory:
                    param_state['memory'] = param_state['memory'] + d_p - corrected_gradient

                p.data.add_(-1, corrected_gradient)

        return loss

    def memory_norm(self):
        """
        :return: The L2 norm of the memory (if any)
        """
        norm = 0
        for group in self.param_groups:
            for p in group['params']:
                n = p.norm()
                norm += float(n * n)
        return np.sqrt(norm)

    def gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['gradient'].norm(p=1)
                n2 = param_state['gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        ''' Correct ratio = (sum of (n1)^2/d)/(sum of (n2)^2).
            The last coordinate of res has the correct ratio. '''
        res.append(sum_normalized_l1_norm/sum_l2_norms)
        return np.array(res)

    def corrected_gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['corrected_gradient'].norm(p=1)
                n2 = param_state['corrected_gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        ''' Correct ratio = (sum of (n1)^2/d)/(sum of (n2)^2).
            The last coordinate of res has the correct ratio. '''
        res.append(sum_normalized_l1_norm/sum_l2_norms)
        return np.array(res)

    def params_dims(self):
        res = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                d = param_state['dim']
                res.append(d)
        return np.array(res)
