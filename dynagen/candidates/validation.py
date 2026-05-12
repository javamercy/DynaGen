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
    return _validate_generated_code(code, contract="tsp")


def validate_bbob_generated_code(code: str) -> ValidationResult:
    return _validate_generated_code(code, contract="bbob")


def validate_dvrp_generated_code(code: str) -> ValidationResult:
    return _validate_generated_code(code, contract="dvrp")


def _validate_generated_code(code: str, *, contract: str) -> ValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exception:
        return ValidationResult(False, f"SyntaxError: {exception}")

    result = _validate_ast(tree, contract=contract)
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


def _validate_ast(tree: ast.AST, *, contract: str) -> ValidationResult:
    solver_node: ast.FunctionDef | None = None
    dvrp_policy_node: ast.FunctionDef | None = None
    optimizer_node: ast.ClassDef | None = None
    top_level_allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)
    if contract == "bbob":
        top_level_allowed = top_level_allowed + (ast.ClassDef,)
    for node in getattr(tree, "body", []):
        if not isinstance(node, top_level_allowed):
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
        elif isinstance(node, ast.FunctionDef) and node.name == "choose_next_customer":
            dvrp_policy_node = node
        elif isinstance(node, ast.ClassDef) and node.name == "Optimizer":
            optimizer_node = node
        elif isinstance(node, ast.Call):
            result = _validate_call(node)
            if not result.valid:
                return result
    if contract == "bbob":
        if optimizer_node is None:
            return ValidationResult(False, "Missing required Optimizer class")
        return _validate_bbob_optimizer_ast(optimizer_node)
    if contract == "dvrp":
        if dvrp_policy_node is None:
            return ValidationResult(False, "Missing required choose_next_customer function")
        return _validate_dvrp_policy_signature_ast(dvrp_policy_node)
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


def _validate_dvrp_policy_signature_ast(node: ast.FunctionDef) -> ValidationResult:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    expected = [
        "current_position",
        "depot_position",
        "truck_positions",
        "available_customers",
        "current_time",
        "seed",
        "budget",
    ]
    if len(positional) != len(expected) or args.vararg or args.kwonlyargs or args.kwarg:
        return ValidationResult(False, "choose_next_customer must accept exactly seven parameters")
    if [arg.arg for arg in positional] != expected:
        return ValidationResult(False, "choose_next_customer parameters must be current_position, depot_position, truck_positions, available_customers, current_time, seed, budget")
    return ValidationResult(True)


def _validate_bbob_optimizer_ast(node: ast.ClassDef) -> ValidationResult:
    init_node: ast.FunctionDef | None = None
    call_node: ast.FunctionDef | None = None
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_node = item
        elif isinstance(item, ast.FunctionDef) and item.name == "__call__":
            call_node = item
    if init_node is None:
        return ValidationResult(False, "Optimizer must define __init__(self, budget, dim, seed)")
    if call_node is None:
        return ValidationResult(False, "Optimizer must define __call__(self, func)")
    init_result = _validate_method_signature_ast(
        init_node,
        ["self", "budget", "dim", "seed"],
        "Optimizer.__init__ must accept self, budget, dim, seed",
    )
    if not init_result.valid:
        return init_result
    return _validate_method_signature_ast(
        call_node,
        ["self", "func"],
        "Optimizer.__call__ must accept self, func",
    )


def _validate_method_signature_ast(node: ast.FunctionDef, names: list[str], message: str) -> ValidationResult:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if len(positional) != len(names) or args.vararg or args.kwonlyargs or args.kwarg:
        return ValidationResult(False, message)
    if [arg.arg for arg in positional] != names:
        return ValidationResult(False, message)
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
