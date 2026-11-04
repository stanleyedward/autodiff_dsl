import autodiff
import compiler
import _asdl.loma as loma_ir
from extensions.remove_extra_atomics import remove_atomics
from extensions.tbr import make_policy
from extensions.checkpoint import checkpoint_loops
import codegen_slang
from extensions.backends import InoutSlangVisitor
from extensions.privatize_calls import privatize_calls

import reverse_diff

_original_parse = compiler.parser.parse
_origina_diff = autodiff.differentiate
_original_slang_visitor = codegen_slang.SlangCodegenVisitor
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
    for name, f in funcs.items():
        if isinstance(f, loma_ir.FunctionDef):
            if use_privatize:
                f = privatize_calls(f, funcs)
            if use_exclusive:
                f = remove_atomics(f)
            funcs[name] = f 
    last_funcs = funcs
    return funcs

def enable(exclusive=True, tbr=True, checkpoint=True, privatize = False, patch_slang=True):
    global use_exclusive, use_privatize
    use_exclusive = exclusive
    use_privatize = privatize
    reverse_diff.should_push_hook = make_policy if tbr else None
    compiler.parser.parse = parse_with_ext if checkpoint else _original_parse
    autodiff.differentiate = differentiate_with_ext
    codegen_slang.SlangCodegenVisitor = InoutSlangVisitor if patch_slang else _original_slang_visitor

def disable():
    autodiff.differentiate = _origina_diff
    reverse_diff.should_push_hook = None
    compiler.parser.parse = _original_parse
    codegen_slang.SlangCodegenVisitor = _original_slang_visitor
