
# class only used for separate namespace
class PropertyChecker:

    @staticmethod
    def check_type(key, val, type_) -> None:
        if not isinstance(val, type_):
            types = str(type_).replace("|", "or")
            raise TypeError(f"Property '{key}' needs to be of type(s) {types}, but is {type(val)}.")

    @staticmethod
    def check_not_above(key, val, cmp) -> None:
        if val > cmp:
            raise ValueError(f"Property '{key}' needs to be below or equal to {cmp}, but is {val}.")

    @staticmethod
    def check_not_below(key, val, cmp) -> None:
        if val < cmp:
            raise ValueError(f"Property '{key}' needs to be above or equal to {cmp}, but is {val}.")

    @staticmethod
    def check_above(key, val, cmp) -> None:
        if val <= cmp:
            raise ValueError(f"Property '{key}' needs to be above {cmp}, but is {val}.")

    @staticmethod
    def check_below(key, val, cmp) -> None:
        if val >= cmp:
            raise ValueError(f"Property '{key}' needs to be below {cmp}, but is {val}.")

    @staticmethod
    def check_callable(key, val) -> None:
        if not callable(val):
            raise TypeError(f"{key} needs to be callable, but is {type(val)}.")
    
    @staticmethod
    def check_none_or_callable(key, val) -> None:
        if val is not None and not callable(val):
            raise TypeError(f"{key} needs to be callable or None, but is '{type(val)}'.")
    
    @staticmethod
    def check_if_element(key, val, list_) -> None:
        if val not in list_:
            raise ValueError(f"Invalid value '{val}' for property '{key}'. Needs to be one of {list_}, but is '{val}'.")

