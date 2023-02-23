from __future__ import annotations
import itertools
import torch

from .. import VMP_OPTIONS
from ..distributions import Distribution, Unit, Normal, MultivariateNormal, Probability
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..factors import Factor


class Variable:
    new_id = itertools.count()
    instance = dict()

    def __init__(self, dim, name: str = "Variable", **kw):
        self._name = name
        self._dim = torch.Size(dim)
        self.id = next(Variable.new_id)
        self.parents: Dict[int, Factor] = {}
        self.children: Dict[int, Factor] = {}
        self.posterior: Distribution = Normal.unit_from_dimension(dim)
        self.samples: torch.Tensor = torch.zeros(1, *self._dim)
        Variable.instance[self.id] = self
        if VMP_OPTIONS["logging"]: print(f"Initialized {repr(self)}")

    def set_parents(self, **kwargs: Factor):
        for k, v in kwargs.items():
            self.parents[v.id] = v

    def set_children(self, **kwargs: Factor):
        for k, v in kwargs.items():
            self.children[v.id] = v

    @property
    def shape(self):
        return self._dim

    def compute_posterior(self):
        post = None
        for parent in self.parents.values():
            if post is None:
                post = parent.messages_to_children[self.id].message_to_variable
            else:
                post *= parent.messages_to_children[self.id].message_to_variable
        for child in self.children.values():
            if post is None:
                post = child.messages_to_parents[self.id].message_to_variable
            else:
                post *= child.messages_to_parents[self.id].message_to_variable
        self.posterior = post

    def update(self, prev_msg, new_msg):
        if VMP_OPTIONS["logging"]: print(f"Update posterior of {self}")
        post = self.posterior * (new_msg / prev_msg)
        if post is None:
            raise ValueError("Posterior is None")
        self.posterior = post

    def __repr__(self):
        return f"[v{self.id}] {self._name}"

    def sample(self, n_samples):
        self.samples = self.posterior.sample(n_samples)
        return self.samples

    @property
    def name(self):
        return self._name


class MultivariateNormalVariable(Variable):

    def __init__(self, dim, name: str = "MultivariateNormalVariable", **kw):
        super(MultivariateNormalVariable, self).__init__(dim, name, **kw)
        self.posterior: Distribution = MultivariateNormal.unit_from_dimension(dim)


class ProbabilityVariable(Variable):

    def __init__(self, dim, name: str = "ProbabilityVariable", **kw):
        super(ProbabilityVariable, self).__init__(dim, name, **kw)
        self.posterior: Distribution = Probability.unit_from_dimension(dim)

class ObservedVariable(Variable):

    def __init__(self, values, name: str = "ObservedVariable", dist: Distribution = Normal, **kw):
        super(ObservedVariable, self).__init__(values.shape, name, **kw)
        self.values = values
        self.samples = values.reshape(1, *self._dim)
        self.posterior: Distribution = dist.unit_from_dimension(self._dim)

    def sample(self, n_samples):
        pass

