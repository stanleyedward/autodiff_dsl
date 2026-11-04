import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import codegen_slang


class InoutSlangVisitor(codegen_slang.SlangCodegenVisitor):
    def visit_function_def(self, node):
        if node.is_simd:
            return super().visit_function_def(node)
        params = []
        for a in node.args:
            if a.i == loma_ir.Out() and not isinstance(a.t, loma_ir.Array):
                params.append(f'inout {codegen_slang.type_to_string(a.t)} {a.id}')
            else:
                params.append(f'{codegen_slang.type_to_string(a)} {a.id}')
        self.code += f'{codegen_slang.type_to_string(node.ret_type)} {node.id}('
        self.code += ', '.join(params) + ') {\n'
        self.byref_args = set()
        self.single_ele_buffer = set()
        self.tab_count += 1
        for stmt in node.body:
            self.visit_stmt(stmt)
        self.tab_count -= 1
        self.emit_tabs()
        self.code += '}\n'

    def visit_expr(self, node):
        if isinstance(node, loma_ir.Call) and node.id in self.func_defs:
            return f'{node.id}(' + ','.join(self.visit_expr(a) for a in node.args) + ')'
        return super().visit_expr(node)
