"""Graft meta-optimizer implementation.
"""

import math
import torch

class Graft(torch.optim.Optimizer):
    """Grafted meta-optimizer for disentanglement of optimizers and
    implicit step size schedules. Takes black-box optimizers M and D,
    and grafts the norm of M's update with the normalized step
    direction of D's update, with in-place operations.
    Also known as AdaGraft.

    Paper: Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren,
           Cyril Zhang. Disentangling Adaptive Gradient Methods from
           Learning Rates. https://arxiv.org/abs/2002.11803

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups; should match child optimizers'
            params for intended usage
        magnitude_optimizer (torch.nn.Optimizer): child optimizer to
            inherit step sizes
        direction_optimizer (torch.nn.Optimizer): child optimizer to
            inherit step directions
        lr (float, optional): global lr multiplier (default: 1.0)
        eps (float, optional): term added to D normalization
            denominator for numerical stability (default: 1e-16)
        use_global_norm (bool, optional): graft global l2 norms rather
            than per-layer (default: False)
    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, params, magnitude_optimizer, direction_optimizer,
                 lr=1.0, eps=1e-16, use_global_norm=False):

        self.magnitude_optimizer = magnitude_optimizer
        self.direction_optimizer = direction_optimizer
        self.use_global_norm = use_global_norm

        self.global_M_norm = self.global_D_norm = None

        defaults = dict(lr=lr, eps=eps)

        super(Graft, self).__init__(params, defaults)

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

        self._increment_step()

        # grafting can be cheaper/stabler when M & D are SGD-like
        self._save_scratch_copy()
        self._step_M_inplace()
        self._step_D_inplace()

        return loss

    def _increment_step(self):
        # possibly initialize step state, and increment it
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0 or 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

    def _save_scratch_copy(self):
        # possibly initialize scratch space, and copy current model weights to it
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0 or 'scratch' not in state:
                    state['scratch'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['scratch'].copy_(p)

    # pylint: disable-msg=invalid-name
    def _step_M_inplace(self):
        # execute M, measure per-layer step norms, then revert step

        # call M in-place
        self.magnitude_optimizer.step()

        # measure M's step norms, then undo M's step
        squared_step_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # stingy/trashy: use p to store step
                p -= state['scratch']
                state['m_norm'] = torch.linalg.norm(p)
                if self.use_global_norm:
                    squared_step_norm += state['m_norm'].item() ** 2

                # revert p to old weights for D
                p.copy_(state['scratch'])

        if self.use_global_norm:
            self.global_M_norm = math.sqrt(squared_step_norm)

    # pylint: disable-msg=invalid-name
    def _step_D_inplace(self):
        # execute D, then rescale step norms measured from M

        # call D in-place
        self.direction_optimizer.step()

        # measure D's step norms
        squared_step_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # use p to store step again
                p -= state['scratch']
                state['d_norm'] = torch.linalg.norm(p)
                if self.use_global_norm:
                    squared_step_norm += state['d_norm'].item() ** 2

        if self.use_global_norm:
            self.global_D_norm = math.sqrt(squared_step_norm)

        # rescale D's step by M's norms
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # rescale p, which is currently step
                if self.use_global_norm:
                    rescale_factor = group['lr'] \
                        * self.global_M_norm / (self.global_D_norm + group['eps'])
                else:
                    rescale_factor = group['lr'] \
                        * state['m_norm'] / (state['d_norm'] + group['eps'])

                # new weights = rescaled step + old copy
                p *= rescale_factor
                p += state['scratch']
