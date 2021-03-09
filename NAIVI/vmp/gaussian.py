import torch


class Gaussian:

    uniform_precision = 1.0e-20
    _min_precision = 1.0e-16

    point_mass_precision = 1.0e+20
    _max_precision = 1.0e+16

    def __init__(self, p, mtp):
        self._precision = None
        self._mean_times_precision = None
        self.set_natural(p, mtp)

    def set_natural(self, p, mtp):
        self._precision = p
        self._mean_times_precision = mtp

    @classmethod
    def from_array_natural(cls, p, mtp):
        return cls(p, mtp)

    @classmethod
    def from_array(cls, m, v):
        p = (1. / v).clamp(cls.uniform_precision, cls.point_mass_precision)
        mtp = m * p
        return cls(p, mtp)

    @classmethod
    def from_shape(cls, shape, m, v):
        return cls.from_array(torch.full(shape, m), torch.full(shape, v))

    @classmethod
    def from_shape_natural(cls, shape, p, mtp):
        return cls.from_array_natural(torch.full(shape, p), torch.full(shape, mtp))

    @classmethod
    def uniform(cls, shape):
        return cls.from_shape_natural(shape, cls.uniform_precision, 0.)

    @classmethod
    def observed(cls, value):
        # allows missing values
        mask = torch.isnan(value)
        m = torch.where(mask, torch.zeros_like(value), value)
        v = torch.where(
            mask,
            torch.full_like(value, cls.point_mass_precision),
            torch.full_like(value, cls.uniform_precision)
        )
        return cls.from_array(m, v)

    @classmethod
    def point_mass(cls, point):
        m = point
        v = torch.full_like(m, 1. / cls.point_mass_precision)
        return cls.from_array(m, v)

    @property
    def device(self):
        return self._precision.device

    def cuda(self):
        self._precision = self._precision.cuda()
        self._mean_times_precision = self._mean_times_precision.cuda()
        return self

    def to(self, device):
        self._precision = self._precision.to(device)
        self._mean_times_precision = self._mean_times_precision.to(device)
        return self

    @property
    def shape(self):
        return self._precision.shape

    def size(self):
        return self._precision.size()

    @property
    def dtype(self):
        return self._precision.dtype

    @property
    def is_point_mass(self):
        return self._precision >= Gaussian._max_precision

    @property
    def is_uniform(self):
        return self._precision <= Gaussian._min_precision

    @property
    def precision(self):
        return self._precision

    @property
    def mean_times_precision(self):
        return self._mean_times_precision

    @property
    def natural(self):
        return self._precision, self._mean_times_precision

    @property
    def mean(self):
        return self._mean_times_precision / self._precision

    @property
    def variance(self):
        return 1. / self._precision

    @property
    def mean_and_variance(self):
        return self.mean, self.variance

    def product(self, dim, keepdim=False):
        return Gaussian(
            torch.nansum(self._precision, dim, keepdim=keepdim),
            torch.nansum(self._mean_times_precision, dim, keepdim=keepdim)
        )

    def set_mean_and_variance(self, m, v):
        p = 1. / v
        mtp = m * p
        self.set_natural(p, mtp)

    def set_to(self, x):
        self.set_natural(*x.natural)

    def negative_entropy(self):
        return - self.entropy()

    def entropy(self):
        entropy = self.variance.log() * 0.5
        entropy = torch.where(
            torch.logical_or(self.is_point_mass, self.is_uniform),
            torch.zeros_like(entropy),
            entropy
        )
        return torch.sum(entropy)

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "Gaussian{}\n".format(tuple(self.shape))
        out += "Mean=\n" + str(self.mean) + "\n"
        out += "Variance=\n" + str(self.variance)
        return out

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if not isinstance(other, Gaussian):
            raise TypeError("other should be a Gaussian")
        p0, mtp0 = self.natural
        p1, mtp1 = other.natural
        return Gaussian(p0+p1, mtp0+mtp1)

    def __truediv__(self, other):
        if not isinstance(other, Gaussian):
            raise TypeError("other should be a Gaussian")
        p0, mtp0 = self.natural
        p1, mtp1 = other.natural
        p = torch.where(other.is_uniform, p0, p0 - p1)
        p = p.clamp(Gaussian.uniform_precision, Gaussian.point_mass_precision)
        mtp = torch.where(other.is_uniform, mtp0, mtp0 - mtp1)
        mtp = torch.where(p<=Gaussian._min_precision, torch.zeros_like(mtp), mtp)
        return Gaussian(p, mtp)

    def __getitem__(self, item):
        p = self._precision[item]
        mtp = self._mean_times_precision[item]
        return Gaussian(p, mtp)

    def __setitem__(self, key, value):
        self._precision[key] = value.precision
        self._mean_times_precision[key] = value.mean_times_precision

    def __imul__(self, other):
        self.set_to(self*other)
        return self

    def __idiv__(self, other):
        self.set_to(self/other)
        return self

    def update(self, old, new):
        self.set_to(self * new / old)

    # tensor manipulation

    def split(self, split_size_or_sections, dim=0):
        ps = self._precision.split(split_size_or_sections, dim)
        mtps = self._mean_times_precision.split(split_size_or_sections, dim)
        return tuple(Gaussian(p, mtp) for p, mtp in zip(ps, mtps))

    @classmethod
    def cat(cls, gaussians, dim=0):
        p = torch.cat([g.precision for g in gaussians], dim)
        mtp = torch.cat([g.mean_times_precision for g in gaussians], dim)
        return Gaussian(p, mtp)

    def squeeze(self, dim=None):
        p = self._precision.squeeze(dim)
        mtp = self._mean_times_precision.squeeze(dim)
        return Gaussian(p, mtp)

    def squeeze_(self, dim=None):
        self._precision.squeeze_(dim)
        self._mean_times_precision.squeeze_(dim)

    def unsqueeze(self, dim):
        p = self._precision.unsqueeze(dim)
        mtp = self._mean_times_precision.unsqueeze(dim)
        return Gaussian(p, mtp)

    def unsqueeze_(self, dim):
        self._precision.unsqueeze_(dim)
        self._mean_times_precision.unsqueeze_(dim)

    def expand(self, *sizes):
        p = self._precision.expand(*sizes)
        mtp = self._mean_times_precision.expand(*sizes)
        return Gaussian(p, mtp)

