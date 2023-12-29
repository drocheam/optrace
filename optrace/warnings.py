import warnings

from . import global_options


class OptraceWarning(UserWarning):
    """Warnings subclass so warnings from this library can be filtered separately with the warnings library"""
    pass

def simplified_warning(message, category, filename, lineno, file=None, line=None) -> str:
    """simplified warning formatting without filename and linenumber as would be default"""
    return "Warning: " + str(message) + "\n" 


def warning(text: str) -> None:
    """emit a warning in a custom warning class with custom formatter and only if global_options allow it"""
       
    if global_options.show_warnings:

        # backup old warning formatter
        formatwarning_org = warnings.formatwarning

        # set new one
        warnings.formatwarning = simplified_warning
        
        # emit warning
        warnings.warn(text, OptraceWarning, stacklevel=2)
        
        # restore old formatter
        warnings.formatwarning = formatwarning_org
