
import copy
import numpy as np

class BaseClass:

    _new_lock = False
    _lock = False
    
    def crepr(self):
        """ Compact state representation using only lists and immutable types """
        d = {x: self.__dict__[x] for x in self.__dict__ if not x.startswith("_")}

        cl = []

        for key in d:

            match d[key]:

                case list():
                    cl.append(tuple(d[key]))

                case np.ndarray():
                    cl.append(id(d[key]))

                case _ if issubclass(type(d[key]), BaseClass):
                    cl.append(d[key].crepr())

                case _ if callable(d[key]):
                    cl.append(id(d[key]))

                case _:
                    cl.append(d[key])

        return cl

    def __str__(self):
        """gets called when print(obj) or repr(obj) gets called"""
        d = {x: self.__dict__[x] for x in self.__dict__ if not x.startswith("_")}
        return f"{self.__class__.__name__} at {hex(id(self))} with {d}"

    def copy(self):
        """Return a fully independent copy"""
        return copy.deepcopy(self)
    
    def lock(self):
        """make storage read only"""

        for key in self.__dict__:
            if isinstance(self.__dict__[key], np.ndarray):
                self.__dict__[key].flags.writeable = False

        self._lock = True

    def __setattr__(self, key, val0):
      
        # work on copies of ndarray and list
        val = val0.copy() if isinstance(val0, list | np.ndarray) else val0
        
        if self._new_lock and key not in self.__dict__:
            raise ValueError(f"Invalid property {key}")
        
        if self._lock and key != "_lock":
            raise RuntimeError("Object is currently read-only. Create a new object with new properties "
                               "or use class methods to change its properties.")
       
        self.__dict__[key] = val

