import ir

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
