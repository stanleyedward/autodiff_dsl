"""
python show_ir.py file.py | show source-level IR, parsed and type-checked
python show_ir.py file.py --tree  | also print each function as an indented node tree
python show_ir.py file.py --diff   | also run autodiff and print the derivative functions
"""
import sys
import parser, autodiff, check, pretty_print
import _asdl.loma as loma_ir

def tree(node, indent=0):
    pad = '  ' * indent
    if isinstance(node, (list, tuple)):
        for n in node:
            tree(n, indent)
        return
    if not hasattr(node, '__attrs_attrs__'):
        print(f'{pad}{node!r}')
        return
    t = getattr(node, 't', None)
    label = type(node).__name__
    if t is not None and isinstance(node, loma_ir.expr):
        label += f'   : {pretty_print.type_to_string(t)}'
    print(pad + label)
    for field in node.__attrs_attrs__:
        if field.name == 'lineno':
            continue
        if field.name == 't' and isinstance(node, loma_ir.expr):
            continue
        value = getattr(node, field.name)
        if hasattr(value, '__attrs_attrs__') or isinstance(value, (list, tuple)):
            print(f'{pad}  .{field.name}:')
            tree(value, indent + 2)
        else:
            print(f'{pad}  .{field.name} = {value!r}')

path = sys.argv[1]
show_tree = '--tree' in sys.argv
show_diff = '--diff' in sys.argv

with open(path) as f:
    structs, funcs = parser.parse(f.read())
structs, diff_structs, funcs = autodiff.resolve_diff_types(structs, funcs)
check.check_ir(structs, diff_structs, funcs, check_diff=False)

print('=' * 20, 'type-checked source', '=' * 20)
for f in funcs.values():
    print(pretty_print.loma_to_str(f))
    if show_tree and isinstance(f, loma_ir.FunctionDef):
        tree(f)
    print()

if show_diff:
    funcs = autodiff.differentiate(structs, diff_structs, funcs)
    if show_tree:
        print('=' * 20, 'derivative trees', '=' * 20)
        for f in funcs.values():
            if f.id.startswith(('d_', '_d_', 'fwd_', 'rev_')):
                print(f'--- {f.id}')
                tree(f)
