"""Ada optimizer: unifying the most common set of optimizers
"""

import math
import torch

class Ada(torch.optim.Optimizer):
    """Generic diagonal preconditioning optimizer.

    It has been noted by many papers that the most commonly-used
    optimizers can be unified by a common framework.
    - https://arxiv.org/abs/1705.08292
    - https://arxiv.org/abs/1910.05446
    - https://arxiv.org/abs/1806.02958

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups
        lr (float, optional): global lr multiplier (default: 1e-3)
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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, adam_scale=False, decouple_weight_decay=False,
                 nesterov=False):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        adam_scale=adam_scale, decouple_weight_decay=decouple_weight_decay,
                        nesterov=nesterov)

        super(Ada, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

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
                    state['m2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1, beta2 = group['betas']
                m1, m2 = state['m1'], state['m2']

                adam_scale = group['adam_scale']
                t = state['step']
                weight_decay = group['weight_decay']
                decouple_weight_decay = group['decouple_weight_decay']
                lr = group['lr']
                grad = p.grad

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.mul_(1 - lr*weight_decay)
                    else:
                        # modify grad non-destructively
                        grad = grad.add(p, alpha=weight_decay)

                # 1st moment (momentum)
                m1 *= beta1
                if adam_scale:
                    m1.add_(grad, alpha=1-beta1)
                else:
                    m1 += grad

                # 2nd moment (adaptive preconditioner)
                m2 *= beta2
                if adam_scale:
                    m2.addcmul_(grad, grad, value=1-beta2)
                else:
                    m2.addcmul_(grad, grad)

                # preconditioned step
                if adam_scale:
                    denom = m2.sqrt().div_(math.sqrt(1 - beta2**(t+1))).add_(group['eps'])
                    p.addcdiv_(m1, denom, value=-lr / (1 - beta1**(t+1)))
                else:
                    denom = m2.sqrt().add_(group['eps'])
                    p.addcdiv_(m1, denom, value=-lr)

                state['step'] += 1

        return loss
