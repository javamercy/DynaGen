import ast
import inspect
from dataclasses import dataclass

ALLOWED_IMPORTS = {"numpy", "math", "random", "heapq", "itertools", "collections", "time"}

UNSAFE_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
}

UNSAFE_ATTRIBUTES = {
    "load",
    "save",
    "savetxt",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "system",
    "popen",
    "spawn",
    "fork",
    "unlink",
    "rmdir",
    "mkdir",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str | None = None


def validate_generated_code(code: str) -> ValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exception:
        return ValidationResult(False, f"SyntaxError: {exception}")

    result = _validate_ast(tree)
    if not result.valid:
        return result
    try:
        compile(code, "<generated_candidate>", "exec")
    except SyntaxError as exception:
        return ValidationResult(False, f"SyntaxError: {exception}")
    return ValidationResult(True)


def validate_solver_signature(func) -> ValidationResult:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        return ValidationResult(False, f"Invalid solve_tsp signature: {exc}")
    params = list(signature.parameters.values())
    if len(params) != 3:
        return ValidationResult(False, "solve_tsp must accept exactly three parameters")
    if [param.name for param in params] != ["distance_matrix", "seed", "budget"]:
        return ValidationResult(False, "solve_tsp parameters must be distance_matrix, seed, budget")
    for param in params:
        if param.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY}:
            return ValidationResult(False, "solve_tsp parameters must be positional")
    return ValidationResult(True)


def _validate_ast(tree: ast.AST) -> ValidationResult:
    solver_node: ast.FunctionDef | None = None
    for node in getattr(tree, "body", []):
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            return ValidationResult(False, f"Top-level {type(node).__name__} statements are not allowed")
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            return ValidationResult(False, "Only docstrings/constants are allowed as top-level expressions")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result = _validate_import(node)
            if not result.valid:
                return result
        elif isinstance(node, ast.FunctionDef) and node.name == "solve_tsp":
            solver_node = node
        elif isinstance(node, ast.Call):
            result = _validate_call(node)
            if not result.valid:
                return result
    if solver_node is None:
        return ValidationResult(False, "Missing required solve_tsp function")
    return _validate_solver_signature_ast(solver_node)


def _validate_solver_signature_ast(node: ast.FunctionDef) -> ValidationResult:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if len(positional) != 3 or args.vararg or args.kwonlyargs or args.kwarg:
        return ValidationResult(False, "solve_tsp must accept exactly three parameters")
    if [arg.arg for arg in positional] != ["distance_matrix", "seed", "budget"]:
        return ValidationResult(False, "solve_tsp parameters must be distance_matrix, seed, budget")
    return ValidationResult(True)


def _validate_import(node: ast.Import | ast.ImportFrom) -> ValidationResult:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        if node.level != 0:
            return ValidationResult(False, "Relative imports are not allowed")
        names = [node.module or ""]
    for name in names:
        root = name.split(".", 1)[0]
        if root not in ALLOWED_IMPORTS:
            return ValidationResult(False, f"Import not allowed: {name}")
    return ValidationResult(True)


def _validate_call(node: ast.Call) -> ValidationResult:
    func = node.func
    if isinstance(func, ast.Name) and func.id in UNSAFE_CALLS:
        return ValidationResult(False, f"Unsafe call not allowed: {func.id}")
    if isinstance(func, ast.Attribute) and func.attr in UNSAFE_ATTRIBUTES:
        return ValidationResult(False, f"Unsafe attribute call not allowed: {func.attr}")
    return ValidationResult(True)
