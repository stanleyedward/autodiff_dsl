from yapf_third_party._ylib2to3.pgen2.tokenize import Funny

import ir
import irmutator
import irvisitor

ir.generate_asdl_file()
import _asdl.loma as loma_ir


def base_var(node):
    match node:
        case loma_ir.Var():
            return node
        case loma_ir.StructAccess():
            return base_var(node.struct)
        case loma_ir.ArrayAccess():
            return base_var(node.array)
        case _:
            assert False


# get thread_id vars
class ThreadIdFinder(irvisitor.IRVisitor):
    def __init__(self):
        self.declared_tid = set()
        self.assigned = set()

    def visit_declare(self, node):
        if isinstance(node.val, loma_ir.Call) and node.val.id == "thread_id":
            self.declared_tid.add(node.target)

    def visit_assign(self, node):
        self.assigned.add(base_var(node.target).id)


def thread_id_vars(func):
    finder = ThreadIdFinder()
    finder.visit_function(func)
    return finder.declared_tid - finder.assigned

# which arrays only ever use arr[i]
class BadUseFinder(irvisitor.IRVisitor):
    def __init__(self, tids):
        self.tids = tids
        self.bad = set()

    def is_tid(self, node):
        if isinstance(node, loma_ir.Var):
            return node.id in self.tids
        if isinstance(node, loma_ir.Call):
            return node.id == "thread_id"
        return False

    def visit_array_access(self, node):
        if isinstance(node.array, loma_ir.Var):  # a[i]
            if not self.is_tid(node.index):
                self.bad.add(node.array.id)
            self.visit_expr(node.index)

        else:  # s.z[i]
            self.visit_expr(node.array)
            self.visit_expr(node.index)

    def visit_var(self, node):
        self.bad.add(node.id)

    def visit_assign(self, node):  # stmt
        self.visit_expr(node.target)
        self.visit_expr(node.val)


def exclusive_arrays(func):
    tids = thread_id_vars(func)
    finder = BadUseFinder(tids)
    finder.visit_function(func)
    arrays = {a.id for a in func.args if isinstance(a.t, loma_ir.Array)}
    return arrays - finder.bad
