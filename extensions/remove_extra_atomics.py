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
