import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from NNVI.vmp.bernoulli import Bernoulli
from NNVI.vmp.gaussian import Gaussian
import NNVI.vmp.factors as f


class CovariateModel(f.VMPFactor, nn.Module):

    def __init__(
            self,
            positions: Gaussian,
            index: torch.Tensor,
            X_cts: torch.Tensor,
            X_bin: torch.Tensor
    ):
        f.VMPFactor.__init__(self)
        self._deterministic = False
        p_cts = X_cts.shape[1]
        p_bin = X_bin.shape[1]
        p = p_cts + p_bin
        n = index.shape[0]
        K = positions.shape[1]

        self._positions = Gaussian.uniform((n, K))
        self._mean = Gaussian.uniform((n, p))
        self._mean_cts = Gaussian.uniform((n, p_cts))
        self._mean_bin = Gaussian.uniform((n, p_bin))

        self._select = f.Select(positions, index, self._positions)
        self._linear = f.Linear(self._positions, self._mean)
        self._split = f.Split(self._mean, (self._mean_cts, self._mean_bin))
        self._gaussian = f.GaussianFactor.observed(self._mean_cts, X_cts, torch.zeros(p_cts))
        self._logistic = f.Logistic.observed(self._mean_bin, X_bin)

    def __str__(self):
        out = "CovariateModel\n"
        out += str(self._select) + "\n"
        out += str(self._linear) + "\n"
        out += str(self._split) + "\n"
        out += str(self._gaussian) + "\n"
        out += str(self._logistic)
        return out

    def forward(self):
        self._select.forward()
        self._linear.forward()
        self._split.forward()
        self._gaussian.forward()
        self._logistic.forward()

    def backward(self):
        self._logistic.backward()
        self._gaussian.backward()
        self._split.backward()
        self._linear.backward()
        self._select.backward()

    def set_weight_and_bias(self, weight, bias):
        self._linear.weight.data = weight
        self._linear.bias.data = bias.view(-1)

    def to_elbo(self):
        return self._gaussian.to_elbo() + self._logistic.to_elbo()


class AdjacencyModel(f.VMPFactor, nn.Module):

    def __init__(
            self,
            positions: Gaussian,
            heterogeneity: Gaussian,
            indices: Tuple[torch.Tensor],
            links: torch.Tensor
    ):
        f.VMPFactor.__init__(self)
        self._deterministic = False
        n = indices[0].shape[0]
        K = positions.shape[1]

        self._positions = tuple(Gaussian.uniform((n, K)) for _ in range(2))
        self._heterogeneity = tuple(Gaussian.uniform((n, 1)) for _ in range(2))
        self._inner_products = Gaussian.uniform((n, 1))
        self._logits = Gaussian.uniform((n, 1))

        self._select_positions = tuple(
            f.Select(positions, i, p)
            for i, p in zip(indices, self._positions)
        )
        self._select_heterogeneity = tuple(
            f.Select(heterogeneity, i, p)
            for i, p in zip(indices, self._heterogeneity)
        )
        self._inner_product = f.InnerProduct(self._positions, self._inner_products)
        self._sum = f.Sum((self._inner_products, *self._heterogeneity), self._logits)
        self._logistic = f.Logistic.observed(self._logits, links)

    def __str__(self):
        out = "AdjacencyModel\n"
        out += str(self._select_positions) + "\n"
        out += str(self._select_heterogeneity) + "\n"
        out += str(self._inner_product) + "\n"
        out += str(self._sum) + "\n"
        out += str(self._logistic)
        return out

    def forward(self):
        for sel in self._select_positions:
            sel.forward()
        for sel in self._select_heterogeneity:
            sel.forward()
        self._inner_product.forward()
        self._sum.forward()
        self._logistic.forward()

    def backward(self):
        self._logistic.backward()
        self._sum.backward()
        self._inner_product.backward()
        for sel in self._select_heterogeneity:
            sel.backward()
        for sel in self._select_positions:
            sel.backward()

    def to_elbo(self):
        return self._logistic.to_elbo()
