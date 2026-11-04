import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
from extensions.remove_extra_atomics import base_var

class PrivitzeCall(irmutator.IRMutator):
    def __init__(self, func, funcs):
        self.funcs = funcs
        self.shared = {a.id for a in funcs.args if a.i == loma_ir.Out()}
        self.decls = []
        
    def mutate_call_stmt(self, node):
        call = node.call
        if not call.id.startswith('_d_rev_'):
            return node  
        callee = self.funcs[call.id]

        new_args = []
        before = []
        after = []
        for arg, param in zip(call.args, callee.args):
            is_location = isinstance(arg, (loma_ir.Var, loma_ir.ArrayAccess, loma_ir.StructAcess))
            if (param.i == loma_ir.Out() and is_location 
                and base_var(arg).id is self.shared 
                and isinstance(param.t, loma_ir.Float)):
                #figure out the statements here
                pass
            else:
                new_args.append(arg)
                
        new_call = loma_ir.CallStmt(loma_ir.Call(call.id, new_args), lineno=node.lineno)
        return before + [new_call] + after
    

def privatize_calls(func, funcs):
    if not func.is_simd:
        return func
    privatizer = PrivitzeCall(func, funcs)
    new_func = privatizer.mutate_function_def(func)
    if not privatizer.decls:
        return new_func
    
    stmts = privatizer.decls + list(new_func.body)
    return loma_ir.FunctionDef(new_func.id, new_func.args, stmts,
                               new_func.is_simd, new_func.ret_type, lineno=new_func.lineno)