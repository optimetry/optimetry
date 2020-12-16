"""Simple batching meta-optimizer implementation.
"""

import math
import torch

class Batcher(torch.optim.Optimizer):
    """Batchification of an optimizer - averages the optimization steps across multiple
    calls to step() - increasing effective batch size.

    Note: This is a natural extension for SGD. Moment based optimizers need to be thought about potentially.
    TODO: Need to re-look at the way to pipe the learning rate to the inner optimizer

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups; should match child optimizers'
            params for intended usage
        optimizer (torch.nn.Optimizer): the base optimizer to repeat
        repetition (int, optional): number of repetitions (default: 1)
        lr (float, optional): global lr multiplier (default: 1.0)

    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, params, optimizer, repetition=1, lr=1.0):

        self.optimizer = optimizer
        defaults = dict(lr=lr)
        self.repetition = repetition
        super(Graft, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single sub-batch step maintaining a counter.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._increment_step()
        if step%self.repetition==0:
            self._initialize_scratches()
        self._step_inplace()
        return loss

    def _increment_step(self):
        # possibly initialize step state, and increment it
        step = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0 or 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

    def _initialize_scratches(self):
        # possibly initialize scratch+cumulant space, and update it to appropriate values
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0 or 'scratch' not in state:
                    state['scratch'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if len(state) == 0 or 'cumulant' not in state:
                    state['cumulant'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if state['step'] - 1 % self.repetition == 0:
                    state['scratch'].copy_(p)
                    state['cumulant'].zero_()

    # pylint: disable-msg=invalid-name
    def _step_inplace(self):
        # call optimizer in-place
        self.optimizer.step()

        # pipe the learning rate into the optimizer - should play well with an external scheduler
        for s,g in zip(self.param_groups, optim.param_groups):
            g['lr'] = s['lr']

        # update the params and the cumulant
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # stingy/trashy: use p to store step
                p -= state['scratch']
                state['cumulant'] += p
                # revert p to old weights
                p.copy_(state['scratch'])
                state['step'] += 1



