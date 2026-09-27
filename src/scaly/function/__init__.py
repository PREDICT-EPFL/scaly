"""The frontend: Function, the template a decorated body becomes, and ConcreteFunction, one named graph."""

from .model import ConcreteFunction, Function, NotConcrete
from .tree import G, L, Tree

__all__ = ["ConcreteFunction", "Function", "G", "L", "NotConcrete", "Tree"]
