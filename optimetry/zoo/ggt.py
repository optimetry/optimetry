"""GGT optimizer: an efficient full-matrix adaptive gradient method.
"""

import math
import torch

class GGT(torch.optim.Optimizer):
    """GGT full-matrix adaptive optimizer. Uses GPU-friendly operations
    for applying pseudoinverse powers of a low-rank Gram matrix. By
    default, this implementation matches AdaGrad/RMSprop in its other
    features; Adam is possible, but beware the beta2 hyperparameter.

    Paper: Naman Agarwal, Brian Bullins, Xinyi Chen, Elad Hazan,
           Karan Singh, Cyril Zhang, Yi Zhang.
           Efficient Full-Matrix Adaptive Regularization.
           https://arxiv.org/abs/1806.02958

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups
        lr (float, optional): global lr multiplier (default: 1e-3)
        window_size (int, optional): length of history (default: 100)
        betas (Tuple[float, float], optional): coefficients used for
            computing running averages of gradient and its square
            (default: (0.9, 0.999))
        eps (float, optional): truncation threshold for SVD
            (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty)
            (default: 0)
        adam_scale (bool, optional): use Adam scaling rule and bias
            correction (default: True)
    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, params, lr=1e-3, window_size=100, betas=(0.9, 1.), eps=1e-4,
                 weight_decay=0, adam_scale=False, decouple_weight_decay=False):

        defaults = dict(lr=lr, window_size=window_size, betas=betas, eps=eps,
                        weight_decay=weight_decay, adam_scale=adam_scale,
                        decouple_weight_decay=decouple_weight_decay)

        super(GGT, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step. In-place implementation
        of grafting has a 1x model dimension overhead; trades numerical
        stability, speed, and memory consumption for full generality.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    # initialize state
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['Gt'] = torch.zeros([group['window_size'], p.numel()],
                                              dtype=p.dtype, device=p.device)
                    state['GtG'] = torch.zeros([group['window_size'], group['window_size']],
                                               dtype=p.dtype, device=p.device)

                beta1, beta2 = group['betas']
                adam_scale = group['adam_scale']
                t, w = state['step'], group['window_size']
                weight_decay = group['weight_decay']
                lr = group['lr']

                Gt, GtG = state['Gt'], state['GtG']
                m1 = state['m1']

                if weight_decay != 0:
                    if group['decouple_weight_decay']:
                        p.mul_(1 - lr*weight_decay)
                    else:
                        p.grad.add_(p, alpha=weight_decay)

                m1 *= beta1
                if adam_scale:
                    m1.add_(p.grad, alpha=1-beta1)
                else:
                    m1.add_(p.grad)

                # overwrite oldest grad in window buffer
                idx = t % w
                Gt[idx, :].copy_(p.grad.view(-1))

                # update new row and column of small Gram matrix
                row = Gt @ p.grad
                if adam_scale:
                    row *= 1 - beta2
                GtG *= beta2
                GtG[idx, :] = GtG[:, idx] = row

                # get eigendecomposition of small Gram matrix
                eigs, V = torch.symeig(GtG, eigenvectors=True)
                precond_eigs = torch.where(eigs >= group['eps'],
                                           eigs.pow(-1.5), torch.zeros_like(eigs))

                # update by GGT-trick-preconditioned step
                precond_grad = Gt.t() @ (V @ (precond_eigs[:, None] * V.t() @ (Gt @ m1)))
                p.sub_(precond_grad,
                       alpha=lr * math.sqrt(1 - beta2**(t+1)) / (1 - beta1**(t+1))
                       if adam_scale else lr)

                state['step'] += 1

        return loss
