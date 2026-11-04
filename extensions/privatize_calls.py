import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
from extensions.remove_extra_atomics import base_var

class PrivitzeCall(irmutator.IRMutator):
    def __init__(self, func, funcs):
        self.funcs = funcs
        self.shared = {a.id for a in func.args if a.i == loma_ir.Out()}
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
            is_location = isinstance(arg, (loma_ir.Var, loma_ir.ArrayAccess, loma_ir.StructAccess))
            if (param.i == loma_ir.Out() and is_location and base_var(arg).id in self.shared and isinstance(param.t, loma_ir.Float)):
                tmp = loma_ir.Var(f'_ptmp_{len(self.decls)}', t=loma_ir.Float())
                self.decls.append(loma_ir.Declare(tmp.id, loma_ir.Float()))
                before.append(loma_ir.Assign(tmp, loma_ir.ConstFloat(0.0)))
                atomic_call = loma_ir.Call("atomic_add", [arg, tmp])
                after.append(loma_ir.CallStmt(atomic_call))
                new_args.append(tmp)
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