"""Recognize a deliberately small subset of Python used to inspect local data."""

from __future__ import annotations

import ast
from pathlib import Path

_BUILTINS = {
    "print",
    "repr",
    "str",
    "int",
    "float",
    "bool",
    "len",
    "list",
    "dict",
    "tuple",
    "set",
    "sorted",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "range",
    "enumerate",
    "zip",
    "all",
    "any",
}
_METHODS = {
    "get",
    "keys",
    "values",
    "items",
    "strip",
    "lstrip",
    "rstrip",
    "split",
    "splitlines",
    "join",
    "lower",
    "upper",
    "replace",
    "startswith",
    "endswith",
    "count",
    "index",
    "read",
    "readline",
    "readlines",
}


def read_only_python_paths(parts: list[str]) -> tuple[str, ...] | None:
    """Return every literal read path, or None when the script needs assessment.

    No code is executed here. Dynamic calls, imports other than JSON, writes,
    reflection and unknown syntax fall back to the normal approval evaluator.
    The caller must validate the returned paths against its trusted roots.
    """
    if len(parts) != 3 or Path(parts[0]).name not in {"python", "python3"} or parts[1] != "-c":
        return None
    try:
        tree = ast.parse(parts[2])
        checker = _ReadOnlyPython()
        for statement in tree.body:
            checker.statement(statement)
        return tuple(checker.paths)
    except (SyntaxError, ValueError, RecursionError):
        return None


class _ReadOnlyPython:
    def __init__(self) -> None:
        self.names: dict[str, str | None] = {}
        self.json_names: set[str] = set()
        self.paths: list[str] = []

    def bind(self, target: ast.expr, literal: str | None = None) -> None:
        if isinstance(target, ast.Name) and target.id not in _BUILTINS | {"open"} | self.json_names:
            self.names[target.id] = literal
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item)
        else:
            raise ValueError("Unsupported assignment")

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name != "json" or name in self.names or name in _BUILTINS | {"open"}:
                    raise ValueError("Unsupported import")
                self.json_names.add(name)
        elif isinstance(node, ast.Assign):
            self.expression(node.value)
            literal = self.literal(node.value)
            for target in node.targets:
                self.bind(target, literal)
        elif isinstance(node, ast.Expr):
            self.expression(node.value)
        elif isinstance(node, ast.For):
            self.expression(node.iter)
            self.bind(node.target)
            # A value assigned inside a loop must not authorize a later dynamic read.
            assigned = {
                item.id
                for child in node.body
                for item in ast.walk(child)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            }
            for name in assigned:
                self.names[name] = None
            for child in node.body + node.orelse:
                self.statement(child)
            for name in assigned:
                self.names[name] = None
        elif isinstance(node, ast.With):
            for item in node.items:
                self.expression(item.context_expr)
                if item.optional_vars is not None:
                    self.bind(item.optional_vars)
            for child in node.body:
                self.statement(child)
        else:
            raise ValueError("Unsupported statement")

    def literal(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return self.names.get(node.id) if isinstance(node, ast.Name) else None

    def expression(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.Name) and node.id in self.names:
            return
        if isinstance(node, ast.Call):
            self.call(node)
            return
        if isinstance(
            node,
            (
                ast.List,
                ast.Tuple,
                ast.Set,
                ast.Dict,
                ast.Subscript,
                ast.Slice,
                ast.BinOp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.JoinedStr,
                ast.FormattedValue,
            ),
        ):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expression(child)
            return
        raise ValueError("Unsupported expression")

    def call(self, node: ast.Call) -> None:
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("Dynamic keyword arguments")
        keywords = {keyword.arg for keyword in node.keywords}
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "open":
                if not 1 <= len(node.args) <= 2 or not keywords <= {"mode", "encoding", "errors"}:
                    raise ValueError("Unsupported file access")
                path = self.literal(node.args[0])
                mode = self.literal(node.args[1]) if len(node.args) == 2 else "r"
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode = self.literal(keyword.value)
                if path is None or mode not in {"r", "rt", "rb"}:
                    raise ValueError("Not a literal read-only file access")
                self.paths.append(path)
            elif name not in _BUILTINS or (name == "print" and "file" in keywords):
                raise ValueError("Unsupported function")
        elif isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in self.json_names:
                if (
                    node.func.attr not in {"load", "loads", "dumps"}
                    or len(node.args) != 1
                    or not keywords <= {"indent", "ensure_ascii", "sort_keys"}
                ):
                    raise ValueError("Unsupported JSON operation")
            else:
                if node.func.attr not in _METHODS:
                    raise ValueError("Unsupported method")
                self.expression(owner)
        else:
            raise ValueError("Dynamic function")
        for argument in node.args:
            self.expression(argument)
        for keyword in node.keywords:
            self.expression(keyword.value)
