import torch
import numpy as np
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, mean, var):
        out_mean = self.linear(mean)
        out_var = torch.matmul(var, (self.linear.weight ** 2).t())
        return out_mean, out_var

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class AddVariance(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(1, dim))

    def forward(self, mean, var):
        return mean, var + self.log_var.exp()

    def elbo(self, mean, var, y):
        y_masked = y.masked_fill(y.isnan(), 0.)
        v = self.log_var.exp()
        elbo = (var + mean ** 2 - 2. * y_masked * mean + y_masked**2) / v
        elbo += self.log_var + np.log(2. * np.pi)
        elbo.masked_fill(y.isnan(), 0.)
        return -0.5 * torch.nansum(elbo)


class Logistic(nn.Module):

    def __init__(self, p=0):
        super().__init__()
        self.p = p # unused, only for consistency with probit model
        self.gaussian = torch.distributions.normal.Normal(0., 1.)
        # these should be class attributes, but to get correct device they are instance attr
        self._p = nn.Parameter(torch.tensor([[[
            0.003246343272134,
            0.051517477033972,
            0.195077912673858,
            0.315569823632818,
            0.274149576158423,
            0.131076880695470,
            0.027912418727972,
            0.001449567805354
        ]]]), requires_grad=False)
        self._s = nn.Parameter(torch.tensor([[[
            1.365340806296348,
            1.059523971016916,
            0.830791313765644,
            0.650732166639391,
            0.508135425366489,
            0.396313345166341,
            0.308904252267995,
            0.238212616409306
        ]]]), requires_grad=False)

    def forward(self, mean, var):
        mean = mean.unsqueeze(-1)
        var = var.unsqueeze(-1)
        t = torch.sqrt(1. + self._s ** 2 * var)
        smot = mean * self._s / t
        Phi = self.gaussian.cdf(smot)
        proba = torch.sum(self._p * Phi, dim=-1)
        return proba

    def elbo(self, mean, var, y):
        y_masked = y.masked_fill(y.isnan(), 0.)
        t = torch.sqrt(mean**2 + var)
        elbo = nn.LogSigmoid()(t) + (y_masked - 0.5) * mean - 0.5 * t
        elbo.masked_fill(y.isnan(), 0.)
        return torch.nansum(elbo)


class InnerProduct(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m0, v0, m1, v1):
        mean = m0 * m1
        var = m0**2 * v1 + m1**2 * v0 + v0 * v1
        return torch.sum(mean, dim=-1, keepdim=True), torch.sum(var, dim=-1, keepdim=True)


class Sum(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        n = len(args) // 2
        mean = args[0]
        var = args[1]
        for i in range(1, n):
            mean = mean + args[2*i]
            var = var + args[2*i+1]
        return mean, var
