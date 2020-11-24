"""Unit tests for generalized diagonal moment optimizer."""

import copy
import unittest

import torch
import optimetry.zoo

class TestAdaGraft(unittest.TestCase):
    """Ada unit tests."""

    def test_nn(self):  # pylint: disable-msg=too-many-locals
        """Small neural net test."""

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

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer_copy = optimetry.zoo.Ada(net_copy.parameters(), lr=lr, adam_scale=True)

        for _ in range(10):
            optimizer.zero_grad()
            loss = (net(X) - y).square().mean()
            loss.backward()
            optimizer.step()

            optimizer_copy.zero_grad()
            loss_copy = (net_copy(X) - y).square().mean()
            loss_copy.backward()
            optimizer_copy.step()

            # check that trajectories match
            for p, p_copy in zip(net.parameters(), net_copy.parameters()):
                self.assertTrue(torch.allclose(p, p_copy, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
