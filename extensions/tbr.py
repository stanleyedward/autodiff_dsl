import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irvisitor
from extensions.remove_extra_atomics import base_var

def make_policy(func):
    needed = needed_values(func)
    return lambda target: base_var(target).id in needed

class VarCollector(irvisitor.IRVisitor):
    def __init__(self):
        self.names = set()
    def visit_var(self, node):
        self.names.add(node.id)

def vars_in(expr):
    collector = VarCollector()
    collector.visit_expr(expr)
    return collector.names

LINEAR_CALLS = {'int2float', 'float2int', 'thread_id', 'atomic_add'}

class NeededCollector(irvisitor.IRVisitor):
    def __init__(self):
        self.needed = set()

    def visit_mul(self, node):
        self.needed.update(vars_in(node.left))
        self.needed.update(vars_in(node.right))

    def visit_div(self, node):
        self.needed.update(vars_in(node.left))
        self.needed.update(vars_in(node.right))

    def visit_call(self, node):
        if node.id not in LINEAR_CALLS:
            for arg in node.args:
                self.needed.update(vars_in(arg))
        for arg in node.args:
            self.visit_expr(arg)

    def visit_array_access(self, node):
        self.needed.update(vars_in(node.index))
        self.visit_expr(node.array)

    def visit_ifelse(self, node):
        self.needed.update(vars_in(node.cond))
        for stmt in node.then_stmts:
            self.visit_stmt(stmt)
        for stmt in node.else_stmts:
            self.visit_stmt(stmt)

    def visit_while(self, node):
        for stmt in node.body:
            self.visit_stmt(stmt)

    def visit_assign(self, node):
        self.visit_expr(node.target)
        self.visit_expr(node.val)

def needed_values(func):
    need_collector = NeededCollector()
    need_collector.visit_function(func)
    return need_collector.needed
