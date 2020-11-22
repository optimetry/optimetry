'''Unit tests for AdaGraft.'''

import copy
import unittest

import torch
import optimetry

class TestAdaGraft(unittest.TestCase):
    '''AdaGraft unit tests.'''

    def test_1d(self):
        '''Scalar test: gradient descent on f(x) = x**2.'''

        x = 10.
        x_torch = torch.tensor(x, requires_grad=True)
        lr = 0.01

        M = torch.optim.SGD([x_torch], lr=lr)
        D = torch.optim.SGD([x_torch], lr=123*lr)
        optimizer = optimetry.AdaGraft([x_torch], M, D)

        for _ in range(10):
            loss = x_torch**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dx = 2*x
            x -= lr * dx

            self.assertAlmostEqual(x_torch.item(), x, places=4)

    def test_nn(self):  # pylint: disable-msg=too-many-locals
        '''Small neural net test.'''

        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(2, 100),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(100, 100),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(100, 1))

        net_copy = copy.deepcopy(net)

        X = torch.randn(20, 2)
        y = torch.randn(20, 1)
        lr = 0.001

        M = torch.optim.Adam(net.parameters(), lr=lr)
        D = torch.optim.Adam(net.parameters(), lr=123*lr)
        optimizer = optimetry.AdaGraft(net.parameters(), M, D)

        M_copy = torch.optim.Adam(net_copy.parameters(), lr=lr)

        for _ in range(10):
            optimizer.zero_grad()
            loss = (net(X) - y).square().mean()
            loss.backward()
            optimizer.step()

            M_copy.zero_grad()
            loss_copy = (net_copy(X) - y).square().mean()
            loss_copy.backward()
            M_copy.step()

            # check that trajectories match
            for p, p_copy in zip(net.parameters(), net_copy.parameters()):
                max_err = (p - p_copy).abs().max().item()
                self.assertLessEqual(max_err, 1e-6)

if __name__ == '__main__':
    unittest.main()
