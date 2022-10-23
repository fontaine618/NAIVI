from __future__ import annotations
import itertools
import torch

from ..distributions import Distribution, Unit
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..factors import Factor


class Variable:
    _name = "Variable"
    new_id = itertools.count()
    instance = dict()

    def __init__(self, dim, **kw):
        self._dim = torch.Size(dim)
        self.id = next(Variable.new_id)
        self.parents: Dict[int, Factor] = {}
        self.children: Dict[int, Factor] = {}
        self.posterior: Distribution = Unit(dim)
        Variable.instance[self.id] = self

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
        # print(f"Update posterior of {self}")
        self.posterior = new_msg * self.posterior / prev_msg

    def __repr__(self):
        return f"[v{self.id}] {self._name}"


class ObservedVariable(Variable):

    _name = "ObservedVariable"

    def __init__(self, values):
        super(ObservedVariable, self).__init__(values.shape)
        self.values = values

