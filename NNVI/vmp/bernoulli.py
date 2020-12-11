import torch
import torch.nn.functional as F


class Bernoulli:

    _eps = 1.0e-10

    def __init__(self, proba):
        self._proba = None
        self.set_proba(proba)

    def set_proba(self, proba):
        self._proba = proba.clamp(0., 1.)

    @classmethod
    def from_array(cls, proba):
        return cls(proba)

    @classmethod
    def from_array_logit(cls, logit):
        return cls(torch.sigmoid(logit))

    @classmethod
    def uniform(cls, shape):
        return cls.from_array(torch.full(shape, 0.5))

    @classmethod
    def observed(cls, value):
        # allow missing values
        value = torch.where(torch.isnan(value), torch.full_like(value, 0.5), value)
        return cls(value)

    def __str__(self):
        out = "Bernoulli{}\n".format(tuple(self.shape))
        out += "Probabilities=\n" + str(self._proba)
        return out

    def __repr__(self):
        return str(self)

    @property
    def device(self):
        return self._proba.device

    def cuda(self):
        self._proba = self._proba.cuda()
        return self

    def to(self, device):
        self._proba = self._proba.to(device)
        return self

    @property
    def shape(self):
        return self._proba.shape

    def size(self):
        return self._proba.size()

    @property
    def dtype(self):
        return self._proba.dtype

    @property
    def is_point_mass(self):
        return torch.logical_or(
            self._proba <= Bernoulli._eps,
            self._proba >= 1. - Bernoulli._eps
        )

    @property
    def is_uniform(self):
        return self._proba == 0.5

    @property
    def proba(self):
        return self._proba

    @property
    def logit(self):
        return self._proba.logit(eps=Bernoulli._eps)

    def negative_entropy(self):
        return - self.entropy()

    def entropy(self):
        return F.binary_cross_entropy(self._proba, self._proba, reduction="sum")

    def __getitem__(self, item):
        proba = self._proba[item]
        return Bernoulli(proba)

    def __setitem__(self, key, value):
        self._proba[key] = value.proba

    # tensor manipulation

    def split(self, split_size_or_sections, dim=0):
        probas = self._proba.split(split_size_or_sections, dim)
        return tuple(Bernoulli(proba) for proba in probas)

    @classmethod
    def cat(cls, gaussians, dim=0):
        proba = torch.cat([g.proba for g in gaussians], dim)
        return Bernoulli(proba)

    def squeeze(self, dim=None):
        proba = self._proba.squeeze(dim)
        return Bernoulli(proba)

    def squeeze_(self, dim=None):
        self._proba.squeeze_(dim)

    def unsqueeze(self, dim):
        proba = self._proba.unsqueeze(dim)
        return Bernoulli(proba)

    def unsqueeze_(self, dim):
        self._proba.unsqueeze_(dim)

    def expand(self, *sizes):
        proba= self._proba.expand(*sizes)
        return Bernoulli(proba)