
import copy  # deepcopy
import numpy as np  # calculations


class BaseClass:

    def __init__(self, 
                 desc:      str = "", 
                 long_desc: str = "",
                 silent:    bool = False,
                 threading: bool = True):
        """"""
        self.desc        = desc
        self.long_desc   = long_desc
        self.silent      = silent
        self.threading   = threading

    def crepr(self) -> list:
        """ 
        Compact state representation using only lists and immutable types.
        Used to compare states of BaseClass objects.
        np.ndarray is only compared for a size < 20, so either ignore these changes or make the arrays read only.
        """
        # The alternative approach would be to detect changes on the object and set a flag or save a timestamp,
        # unfortunately it would be hard to not only detect changes on the object itself, but as well on all its members,
        # which can also be classes or types such as list or np.ndarray

        d = {x: val for x, val in self.__dict__.items() if not x.startswith("_")}
        cl = []

        for key, val in d.items():

            if isinstance(val, list):                               el = tuple(val)
            elif issubclass(type(val), BaseClass):                  el = val.crepr()
            elif isinstance(val, np.ndarray) and val.size < 20:     el = tuple(val)
            elif isinstance(val, np.ndarray) or callable(val):      el = id(val)
            else:                                                   el = val

            cl.append(el)

        return cl

    def getLongDesc(self, fallback: str="") -> str:
        
        return self.long_desc if self.long_desc != "" else BaseClass.getDesc(self, fallback)

    def getDesc(self, fallback: str="") -> str:
        """"""
        return self.desc if self.desc != "" else fallback
    
    def copy(self) -> 'BaseClass':
        """Return a fully independent copy"""
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

    def __setattr__(self, key, val) -> None:
      
        if key not in ["_lock", "_new_lock"]:
            if "_new_lock" in self.__dict__ and self._new_lock and key not in self.__dict__:
                raise AttributeError(f"Invalid property {key}.")
            
            if "_lock" in self.__dict__ and self._lock:
                raise RuntimeError("Object is currently read-only. Create a new object with new properties "
                                   "or use class methods to change its properties.")
       
        if key in ["desc", "long_desc"]:
            self._checkType(key, val, str)

        if key in ["silent", "threading"]:
            self._checkType(key, val, bool)

        self.__dict__[key] = val

    @staticmethod
    def _checkType(key, val, type_) -> None:
        if not isinstance(val, type_):
            types = str(type_).replace("|", "or")
            raise TypeError(f"Property '{key}' needs to be of type(s) {types}, but is {type(val)}.")

    @staticmethod
    def _checkNotAbove(key, val, cmp) -> None:
        if val > cmp:
            raise ValueError(f"Property '{key}' needs to be below or equal to {cmp}, but is {val}.")

    @staticmethod
    def _checkNotBelow(key, val, cmp) -> None:
        if val < cmp:
            raise ValueError(f"Property '{key}' needs to be above or equal to {cmp}, but is {val}.")

    @staticmethod
    def _checkAbove(key, val, cmp) -> None:
        if val <= cmp:
            raise ValueError(f"Property '{key}' needs to be above {cmp}, but is {val}.")
    
    @staticmethod
    def _checkBelow(key, val, cmp) -> None:
        if val >= cmp:
            raise ValueError(f"Property '{key}' needs to be below {cmp}, but is {val}.")
    
    @staticmethod
    def _checkNoneOrCallable(key, val) -> None:
        if val is not None and not callable(val):
             raise TypeError(f"{key} needs to be callable or None, but is {type(val)}.")
    
    @staticmethod
    def _checkIfIn(key, val, list_) -> None:
        if val not in list_:
            raise ValueError(f"Invalid value '{val}' for property '{key}'. Needs to be one of {list_}, but is {val}.")

