import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irvisitor

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
