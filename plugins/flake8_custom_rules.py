class NoPrintChecker:
    name = "no-print-checker"
    version = "0.1"

    def __init__(self, tree, *args):
        self.tree = tree

    def run(self):
        import ast
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    yield (
                        node.lineno,
                        node.col_offset,
                        "G001 print() statements are not allowed. Use logger.debug() or equivalent.",
                        type(self),
                    )
