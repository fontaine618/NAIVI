import torch


def sigmoid_integrals(mean, variance, degrees=(0, 1, 2)):
    # integrals of the form
    # int_-inf^inf x^r Phi(m+vx)phi(x) dx
    # for r in degrees

    dtype = mean.dtype
    device = mean.device

    _p = torch.tensor([
        0.003246343272134,
        0.051517477033972,
        0.195077912673858,
        0.315569823632818,
        0.274149576158423,
        0.131076880695470,
        0.027912418727972,
        0.001449567805354
    ], dtype=dtype, device=device)

    _s = torch.tensor([
        1.365340806296348,
        1.059523971016916,
        0.830791313765644,
        0.650732166639391,
        0.508135425366489,
        0.396313345166341,
        0.308904252267995,
        0.238212616409306
    ], dtype=dtype, device=device)

    # fix dimensions
    n_dim = len(mean.shape)
    for _ in range(n_dim):
        _p.unsqueeze_(0)
        _s.unsqueeze_(0)
    mean = mean.unsqueeze(-1)
    variance = variance.unsqueeze(-1)

    # integrals
    t = torch.sqrt(1. + _s ** 2 * variance)
    smot = mean * _s / t
    phi = torch.distributions.Normal(0., 1.).log_prob(smot).exp()
    Phi = torch.distributions.Normal(0., 1.).cdf(smot)
    integral = dict()
    if 0 in degrees:
        integral[0] = torch.sum(_p * Phi, -1)
    if 1 in degrees:
        psot = _p * _s / t
        integral[1] = torch.sum(psot * phi, -1) * torch.squeeze(torch.sqrt(variance), -1)
    if 2 in degrees:
        svot = (_s / t) ** 2 * variance
        integral[2] = torch.sum(_p * (Phi + svot * smot * phi), -1)
    return integral