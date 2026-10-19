import autodiff
import _asdl.loma as loma_ir
from extensions.remove_extra_atomics import remove_atomics

_origina_diff = autodiff.differentiate
last_funcs = None
def differentiate_with_ext(structs, diff_structs, funcs):
    global last_funcs
    funcs = _origina_diff(structs, diff_structs, funcs)
    for name, f in funcs.items():
        if isinstance(f, loma_ir.FunctionDef):
            funcs[name] = remove_atomics(f)
    last_funcs = funcs
    return funcs

def enable():
    autodiff.differentiate = differentiate_with_ext

def disable():
    autodiff.differentiate = _origina_diff
