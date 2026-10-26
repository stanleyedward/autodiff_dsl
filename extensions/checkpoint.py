import ir
import irmutator

ir.generate_asdl_file()
import _asdl.loma as loma_ir
from extensions.remove_extra_atomics import base_var
from extensions.tbr import VarCollector, vars_in


class StmtVarCollector(VarCollector):
    # get all variable names in a statment
    def visit_assign(self, node):
        self.visit_expr(node.target)  # parent doesnt visit node.targets
        self.visit_expr(node.val)


def vars_in_stmt(statement):
    collector = StmtVarCollector()
    collector.visit_stmt(statement)
    return collector.names


def reads_of(stmt):
    match stmt:
        case loma_ir.CallStmt():
            return vars_in(stmt.call)
        case loma_ir.Assign():
            var_names = vars_in(stmt.val)
            if not isinstance(stmt.target, loma_ir.Var):
                var_names.update(vars_in(stmt.target))
            return var_names
        case _:  # remaining
            return vars_in_stmt(stmt)


def writes_of(stmt):
    match stmt:
        case loma_ir.Assign():
            return {base_var(stmt.target).id}
        case loma_ir.CallStmt():
            var_names = set()
            for a in stmt.call.args:
                if isinstance(
                    a, (loma_ir.Var, loma_ir.ArrayAccess, loma_ir.StructAccess)
                ):
                    var_names.add(base_var(a).id)
            return var_names
        case loma_ir.IfElse():
            var_names = set()
            for s in stmt.then_stmts + stmt.else_stmts:
                var_names.update(writes_of(s))
            return var_names
        case loma_ir.While():
            var_names = set()
            for s in stmt.body:
                var_names.update(writes_of(s))
            return var_names
        case _:
            return set()


def definite_writes(stmt):
    if isinstance(stmt, loma_ir.Assign) and isinstance(stmt.target, loma_ir.Var):
        return {stmt.target.id}
    return set()


def carried_vars(loop):
    defined, read_first, written = set(), set(), set()
    for s in loop.body:
        read_first.update(reads_of(s) - defined)
        written.update(writes_of(s))
        defined.update(definite_writes(s))
    return written & (read_first | vars_in(loop.cond))


def written_vars(loop):
    names = set()
    for s in loop.body:
        names.update(writes_of(s))
    return names


def read_vars(loop):
    names = set(vars_in(loop.cond))
    for s in loop.body:
        names.update(vars_in_stmt(s))
    return names


def classify_vars(loop):
    state = carried_vars(loop)
    written = written_vars(loop)
    temporaries = written - state
    inputs = read_vars(loop) - written
    return {"state": state, "inputs": inputs, "temps": temporaries}


def mentioned_outside(func, loop):
    names = set()
    def walk(stmts):
        for s in stmts:
            if s is loop:
                continue
            if isinstance(s, loma_ir.Declare):
                if s.val is not None:
                    names.update(vars_in(s.val))
            elif isinstance(s, loma_ir.IfElse):
                names.update(vars_in(s.cond))
                walk(s.then_stmts)
                walk(s.else_stmts)
            elif isinstance(s, loma_ir.While):
                names.update(vars_in(s.cond))
                walk(s.body)
            else:
                names.update(vars_in_stmt(s))

    walk(func.body)
    return names

def var_types_of(func):
    types = {a.id: a.t for a in func.args}
    def walk(stmts):
        for s in stmts:
            if isinstance(s, loma_ir.Declare):
                types[s.target] = s.t
            elif isinstance(s, loma_ir.IfElse):
                walk(s.then_stmts)
                walk(s.else_stmts)
            elif isinstance(s, loma_ir.While):
                walk(s.body)
    walk(func.body)
    return types

def build_segment(seg_name, loop, groups, types):
    state = sorted(groups['state'])
    inputs = sorted(groups['inputs'])
    temps = sorted(groups['temps'])
    k = loop.checkpoint

    for name in state: assert isinstance(types[name], (loma_ir.Int, loma_ir.Float))

    args = []
    for name in state: args.append(loma_ir.Arg(f'{name}_in', types[name], loma_ir.In()))
    for name in inputs: args.append(loma_ir.Arg(name, types[name], loma_ir.In()))
    for name in state: args.append(loma_ir.Arg(f'{name}_out', types[name], loma_ir.Out()))

    body = []
    for name in state: body.append(loma_ir.Declare(name, types[name], loma_ir.Var(f'{name}_in')))
    for name in temps: body.append(loma_ir.Declare(name, types[name]))

    j = loma_ir.Var('_checkpoint_j')
    decl_j = loma_ir.Declare('_checkpoint_j', loma_ir.Int(), loma_ir.ConstInt(0))
    body.append(decl_j)

    cond = loma_ir.BinaryOp(loma_ir.And(), loop.cond, loma_ir.BinaryOp(loma_ir.Less(), j, loma_ir.ConstInt(k)))
    increment = loma_ir.Assign(j, loma_ir.BinaryOp(loma_ir.Add(), j, loma_ir.ConstInt(1)))
    while_stmt = loma_ir.While(cond, k, loop.body + [increment])
    body.append(while_stmt)

    for name in state:
        body.append(loma_ir.Assign(loma_ir.Var(f'{name}_out'), loma_ir.Var(name)))

    return loma_ir.FunctionDef(seg_name, args, body, False, None)

class LoopCheckpointMutator(irmutator.IRMutator):
    def __init__(self, func):
        self.func = func
        self.types = var_types_of(func)
        self.new_funcs = []

    def mutate_while(self, node):
        if node.checkpoint is None:
            return super().mutate_while(node)

        groups = classify_vars(node)
        temps = groups['temps']
        outside_vars = mentioned_outside(self.func, node)
        leaked = temps & outside_vars

        assert not leaked
        assert not any(isinstance(s, loma_ir.While) and s.checkpoint is not None for s in node.body)

        segment_name = f'_checkpoint_{self.func.id}_{len(self.new_funcs)}'
        self.new_funcs.append(build_segment(segment_name, node, groups, self.types))

        state = sorted(groups['state'])
        inputs = sorted(groups['inputs'])

        call_args = [ ]
        call_stmt = loma_ir.CallStmt(loma_ir.Call(segment_name, call_args))
        outer_loop = #notdone
        return outer_loop

def checkpoint_loops(func):
    checkpointer = LoopCheckpointMutator(func)
    checkpointed_func = checkpointer.mutate_function_def(func)
    return checkpointed_func
