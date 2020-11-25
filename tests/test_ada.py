"""Unit tests for generalized diagonal moment optimizer."""

import copy
import unittest

import torch
import optimetry.zoo

def tiny_net():
    """Small deterministic neural net regression problem for tests."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 100),
                              torch.nn.ReLU(),
                              torch.nn.Linear(100, 100),
                              torch.nn.ReLU(),
                              torch.nn.Linear(100, 1))
    X = torch.randn(20, 2)
    y = torch.randn(20, 1)
    return net, X, y

class TestAdaGraft(unittest.TestCase):
    """Ada unit tests."""

    def test_adam(self):
        """Test that Adam matches."""

        net, X, y = tiny_net()
        net_copy = copy.deepcopy(net)

        lr = 1e-2
        weight_decay = 1e-3

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_copy = optimetry.zoo.Ada(net_copy.parameters(), lr=lr,
                                           weight_decay=weight_decay, adam_scale=True)

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
                self.assertTrue(torch.allclose(p, p_copy, atol=1e-6))

    def test_adamw(self):
        """Test that AdamW matches."""

        net, X, y = tiny_net()
        net_copy = copy.deepcopy(net)

        lr = 1e-2
        weight_decay = 1e-3

        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_copy = optimetry.zoo.Ada(net_copy.parameters(), lr=lr,
                                           weight_decay=weight_decay, adam_scale=True,
                                           decouple_weight_decay=True)

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
                self.assertTrue(torch.allclose(p, p_copy, atol=1e-6))

    def test_adagrad(self):
        """Test that AdaGrad matches."""

        net, X, y = tiny_net()
        net_copy = copy.deepcopy(net)

        lr = 1e-2
        weight_decay = 1e-3

        optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_copy = optimetry.zoo.Ada(net_copy.parameters(), lr=lr, betas=(0, 1), eps=1e-10,
                                           weight_decay=weight_decay)

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
                self.assertTrue(torch.allclose(p, p_copy, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
