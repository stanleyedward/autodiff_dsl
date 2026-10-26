import autodiff
import compiler
import _asdl.loma as loma_ir
from extensions.remove_extra_atomics import remove_atomics
from extensions.tbr import make_policy
from extensions.checkpoint import checkpoint_loops

import reverse_diff

_original_parse = compiler.parser.parse
_origina_diff = autodiff.differentiate
last_funcs = None
use_exclusive = True

def parse_with_ext(code):
    structs, funcs = _original_parse(code)
    new_funcs = {}
    for name, f in funcs.items():
        if isinstance(f, loma_ir.FunctionDef):
            rewritten, segments = checkpoint_loops(f)
            for seg in segments:
                new_funcs[seg.id] = seg
            new_funcs[name] = rewritten
        else:
            new_funcs[name] = f
    return structs, new_funcs

def differentiate_with_ext(structs, diff_structs, funcs):
    global last_funcs
    funcs = _origina_diff(structs, diff_structs, funcs)
    if use_exclusive:
        for name, f in funcs.items():
            if isinstance(f, loma_ir.FunctionDef):
                funcs[name] = remove_atomics(f)
    last_funcs = funcs
    return funcs

def enable(exclusive=True, tbr=True, checkpoint=True):
    global use_exclusive
    use_exclusive = exclusive
    reverse_diff.should_push_hook = make_policy if tbr else None
    compiler.parser.parse = parse_with_ext if checkpoint else _original_parse
    autodiff.differentiate = differentiate_with_ext

def disable():
    autodiff.differentiate = _origina_diff
    reverse_diff.should_push_hook = None
    compiler.parser.parse = _original_parse
