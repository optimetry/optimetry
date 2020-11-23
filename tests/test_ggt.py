"""Unit tests for GGT."""

import unittest

import torch
import optimetry.zoo


class TestGGT(unittest.TestCase):
    """GGT unit tests."""

    def test_olo(self):
        """Online linear optimization test for GGT, testing for correctness."""

        def pseudopower(M, alpha, eps):
            # the pseudoinverse power of a psd matrix M
            # inefficient, unlike the low-rank GGT trick; for debugging
            eigs, V = torch.symeig(M, eigenvectors=True)
            pow_eigs = torch.where(eigs >= eps, eigs.pow(alpha), torch.zeros_like(eigs))
            return V @ (pow_eigs[:, None] * V.t())

        torch.manual_seed(0)
        x = torch.zeros(10, requires_grad=True)
        lr = 1.
        grad_gram = torch.zeros((10, 10))

        optimizer = optimetry.zoo.GGT([x], lr, window_size=100, betas=(0, 1))

        for _ in range(20):
            # sample a random linear loss
            g = torch.randn(10)
            loss = g.dot(x)

            # manually compute preconditioned step, inefficiently
            grad_gram += torch.ger(g, g)
            precond_grad = pseudopower(grad_gram, -0.5, 1e-4) @ g
            full_step = x.data - lr * precond_grad

            # take a GGT step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # numerical instability can get sketchy
            self.assertTrue(torch.allclose(x, full_step, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
