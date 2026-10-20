import autodiff
import _asdl.loma as loma_ir
from extensions.remove_extra_atomics import remove_atomics
from extensions.tbr import make_policy
import reverse_diff

_origina_diff = autodiff.differentiate
last_funcs = None
use_exclusive = True

def differentiate_with_ext(structs, diff_structs, funcs):
    global last_funcs
    funcs = _origina_diff(structs, diff_structs, funcs)
    if use_exclusive:
        for name, f in funcs.items():
            if isinstance(f, loma_ir.FunctionDef):
                funcs[name] = remove_atomics(f)
    last_funcs = funcs
    return funcs

def enable(exclusive=True, tbr=True):
    global use_exclusive
    use_exclusive = exclusive
    reverse_diff.should_push_hook = make_policy if tbr else None
    autodiff.differentiate = differentiate_with_ext

def disable():
    autodiff.differentiate = _origina_diff
    reverse_diff.should_push_hook = None
