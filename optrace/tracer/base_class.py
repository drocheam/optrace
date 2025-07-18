import copy  # deepcopy
from typing import Any, Self

import numpy as np  # calculations

from ..property_checker import PropertyChecker as pc  # type checking
import time

class BaseClass:

    def __init__(self,
                 desc:      str = "",
                 long_desc: str = ""):
        """
        common parent class for almost all `optrace.tracer` classes.
        Features description properties as well
        as methods for making the object read-only (kind of)

        :param desc: compact description
        :param long_desc: verbose description
        """
        self._lock = False
        self._new_lock = False

        self.desc = desc
        self.long_desc = long_desc

    def crepr(self) -> list:
        """
        Compact state representation using only lists and immutable types.
        Used to compare states of BaseClass objects.
        np.ndarray is only compared for a size < 20, so either ignore these changes or make the arrays read only.

        :return: compact representation list
        """
        # The alternative approach would be to detect changes on the object and set a flag or save a timestamp,
        # unfortunately it would be hard to not only detect changes on the object itself,
        # but as well on all its members, which can also be classes or types such as list or np.ndarray

        cl = []

        for key, val in self.__dict__.items():

            if issubclass(type(val), BaseClass):
                cl.append(val.crepr())

            elif isinstance(val, np.ndarray):
                cl.append(tuple(val.flat) if val.size < 20 else id(val))

            elif callable(val):
                cl.append(id(val))
            
            elif isinstance(val, list):
                cl.append(tuple(val))

            else:
                cl.append(val)

        return cl

    def get_long_desc(self, fallback: str = "") -> str:
        """
        get a longer, more verbose description
        :param fallback: description string if the object has no desc and long_desc
        :return: verbose description
        """
        return self.long_desc if self.long_desc != "" else self.get_desc(fallback)

    def get_desc(self, fallback: str = "") -> str:
        """
        get a short description
        :param fallback: description string if the object has no desc
        :return: compact description
        """
        return self.desc if self.desc != "" else fallback

    def copy(self) -> Self:
        """:return: a fully independent copy"""
        return copy.deepcopy(self)

    def lock(self) -> None:
        """make storage and object read only"""

        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                val.flags.writeable = False

        self._lock = True
        self._new_lock = True

    def __str__(self) -> str:
        """gets called when print(obj) or repr(obj) gets called"""
        d = {x: val for x, val in self.__dict__.items() if not x.startswith("_")}
        return f"{self.__class__.__name__} at {hex(id(self))} with {d}"

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        if key not in ["_lock", "_new_lock"]:
            if "_new_lock" in self.__dict__ and self._new_lock\
                    and not (key in self.__dict__ or key in type(self).__dict__):
                raise AttributeError(f"Failed to set invalid/unknown property {key}.")

            if "_lock" in self.__dict__ and self._lock and key not in ["desc", "long_desc"]:
                raise RuntimeError("Object is currently read-only. Create a new object with new properties "
                                   "or use class methods to change its properties.")

        if key in ["desc", "long_desc"]:
            pc.check_type(key, val, str)

        self.__dict__[key] = val
