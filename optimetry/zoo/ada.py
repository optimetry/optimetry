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
        gammas (Tuple[float, float], optional): secondary coefficients
            used for computing running averages of moments
            (default: (1-betas[0],1-betas[1]))
        adam_bias_correction (bool, optional): use Adam bias
            correction (default: False)
        decouple_weight_decay (bool, optional): decouple the weight
            decay from the optimizer (default:False)
        condition_before_momentum (bool, optional): perform preconditioning
            before momentum update (default:False)
        nesterov (bool, optional): use Nesterov momentum. Not Implemented.
            (default: False)

    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, gammas=None, adam_bias_correction=False,
                 decouple_weight_decay=False, condition_before_momentum=False, nesterov=False):

        if gammas == None:
            gammas = ((1 - betas[0]), (1-betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gammas=gammas, adam_bias_correction=adam_bias_correction,
                        decouple_weight_decay=decouple_weight_decay, nesterov=nesterov,
                        condition_before_momentum=condition_before_momentum)

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
                gamma1, gamma2 = group['gammas']
                m1, m2 = state['m1'], state['m2']
                t = state['step']
                weight_decay = group['weight_decay']
                decouple_weight_decay = group['decouple_weight_decay']
                condition_before_momentum = group['condition_before_momentum']
                adam_bias_correction = group['adam_bias_correction']
                lr = group['lr']
                grad = p.grad

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.mul_(1 - lr*weight_decay)
                    else:
                        # modify grad non-destructively
                        grad = grad.add(p, alpha=weight_decay)


                # 1st moment (momentum)
                if not condition_before_momentum:
                    m1 *= beta1
                    m1.add_(grad, alpha=gamma1)

                # 2nd moment (adaptive preconditioner)
                m2 *= beta2
                m2.addcmul_(grad, grad, value=gamma2)

                # preconditioned step
                step_direction = grad if condition_before_momentum else m1
                if adam_bias_correction:
                    denom = m2.sqrt().div_(math.sqrt(1 - beta2**(t+1))).add_(group['eps'])
                else:
                    denom = m2.sqrt().add_(group['eps'])
                pre_step = step_direction/denom

                if condition_before_momentum:
                    m1 *= beta1
                    m1.add_(pre_step, alpha=gamma1)
                    pre_step = m1

                if adam_bias_correction:
                    p.add_(pre_step, alpha=-lr/(1 - beta1**(t+1)))
                else:
                    p.add_(pre_step, alpha=-lr)

                state['step'] += 1

        return loss
