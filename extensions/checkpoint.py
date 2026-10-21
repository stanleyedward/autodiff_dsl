import ir
ir.generate_asdl_file()
from extensions.tbr import VarCollector

class StmtVarCollector(VarCollector):
    #get all variable names in a statment
    def visit_assign(self, node):
        self.visit_expr(node.target)#parent doesnt visit node.targets
        self.visit_expr(node.val)

def vars_in_stmt(statement):
    collector = StmtVarCollector()
    collector.visit_stmt(statement)
    return collector.names
