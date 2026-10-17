#i have removed the homework code for my public repo
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):

        def add(self, a, b): return loma_ir.BinaryOp(loma_ir.Add(), a, b)
        def sub(self, a, b): return loma_ir.BinaryOp(loma_ir.Sub(), a, b)
        def mul(self, a, b): return loma_ir.BinaryOp(loma_ir.Mul(), a, b)
        def div(self, a, b): return loma_ir.BinaryOp(loma_ir.Div(), a, b)
        def neg(self, a): return self.sub(loma_ir.ConstFloat(0.0), a)

        def mutate_function_def(self, node):

        def mutate_return(self, node):


        def mutate_declare(self, node):


        def mutate_assign(self, node):


        def mutate_ifelse(self, node):

        def compare(self, node):
            left_val, dval = self.mutate_expr(node.left)
            right_val, dval = self.mutate_expr(node.right)
            return (loma_ir.BinaryOp(node.op, left_val, right_val, lineno=node.lineno), loma_ir.ConstFloat(0.0))

        def mutate_less(self, node):
        def mutate_less_equal(self, node):
        def mutate_greater(self, node):
        def mutate_greater_equal(self, node):
        def mutate_equal(self, node):
        def mutate_and(self, node):
        def mutate_or(self, node):

        def mutate_while(self, node):


        def mutate_const_float(self, node):

        def mutate_const_int(self, node):

        def mutate_var(self, node):

        def mutate_array_access(self, node):

        def mutate_lvalue(self, node):

        def mutate_struct_access(self, node):

        def location_to_pair(self, loc, type):
            match type:
                case loma_ir.Int():
                    return (loc, loma_ir.ConstFloat(0.0))
                case loma_ir.Float():
                    return (loma_ir.StructAccess(loc, 'val'),
                            loma_ir.StructAccess(loc, 'dval'))
                case loma_ir.Struct():
                    return (loc, None)

        def join_pair(self, val, dval, type):
            if isinstance(type, loma_ir.Float):
                return loma_ir.Call('make__dfloat', [val, dval])
            return val

        def mutate_add(self, node):


        def mutate_sub(self, node):

        def mutate_mul(self, node):


        def mutate_div(self, node):

        def mutate_call(self, node):

        def mutate_call_stmt(self, node):

    return FwdDiffMutator().mutate_function_def(func)
