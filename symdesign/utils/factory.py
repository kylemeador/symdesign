from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

SingletonT = TypeVar('SingletonT')
FactoryT = TypeVar('FactoryT')


class SingletonABC(Generic[SingletonT], ABC):

    def __init__(self, **kwargs):
        self._singleton: SingletonT | None = None

    def destruct(self, **kwargs):
        """Dereferences the singleton object."""
        self._singleton = None

    @abstractmethod
    def __call__(self, *args, **kwargs) -> SingletonT:
        ...

    # @Base.factory_call  # ()
    # @classmethod
    # @factory_call  # ()
    # Todo wraps sort of works here, but doesn't use any inheritance
    # @wraps(__call__,
    #        # assigned=("__module__", "__doc__", "__annotations__", "__type_params__"),  # "__name__", "__qualname__",
    #        updated=())
    # @classmethod
    def get(self, *args, **kwargs) -> SingletonT:
        """Returns the factory singleton"""
        if args or kwargs:
            raise ValueError(
                f"{self.__class__.__name__}.{self.get.__name__}() accepts no arguments. Instantiate the singleton by "
                f"calling the factory, such as {self.__class__.__name__}({', '.join(map(repr, args))}, "
                f"{', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())})"
            )

        return self.__call__()


class FactoryABC(Generic[FactoryT], ABC):

    def __init__(self, **kwargs):
        self._instances = {}

    def destruct(self, **kwargs):
        """Dereferences all factory instances."""
        self._instances = {}

    @abstractmethod
    def __call__(self, *args, **kwargs) -> FactoryT:
        ...

    # @Base.factory_call  # ()
    def get(self, *args, **kwargs) -> FactoryT:
        """Returns a factory instance according to provided arguments."""
        return self.__call__(*args, **kwargs)
