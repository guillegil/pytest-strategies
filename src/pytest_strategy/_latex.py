"""
The LaTeX rendering of the export: the ``latex`` of an argument (the set its
values are drawn from) and of a constraint, as math-mode text without ``$``
delimiters::

    RNGInteger(1, 20)                                      addr \\in [1, 20]
    RNGInteger(1, 20, predicate=lambda x: x not in (12, 13))
                                             addr \\in [1, 20],\\ addr \\notin \\{12, 13\\}
    {"fits": lambda v: v.addr + v.len <= 4096}             addr + len \\le 4096

(each name inside ``\\mathit{}``). The bounds, choices and members come from what
the RNG type declares. A predicate or a constraint is a Python function, so its
condition is translated from its source, and only when it is simple: a lambda,
or a function whose body is one ``return``, made of comparisons, arithmetic,
``and``/``or``/``not``, ``in`` and ``abs``/``len``/``min``/``max`` over the
argument, numbers, strings, enum members and names bound to such values.
Anything else is not translated: a predicate is then written ``p(name)``, and a
constraint has no ``latex``. Nothing here raises.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import linecache
import math
import types
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from .rng import (
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
)

# A set with more values than this is cut: its first values, \dots and its last
MAX_LISTED = 8
_LISTED_HEAD = 3

# The longest text (a string value, a repr) written in full
MAX_TEXT = 40

# Integers this large are written as a power of two when they are one (or one less)
_POWER_FROM = 2**16

_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


class _Untranslatable(Exception):
    """The function, or a part of it, is outside what is translated."""


# ---------------------------------------------------------------------------
# Names and values
# ---------------------------------------------------------------------------


def text(value: str) -> str:
    r"""Return ``value`` as ``\text{...}``: cut, one line, LaTeX's special characters escaped."""
    if len(value) > MAX_TEXT:
        value = value[: MAX_TEXT - 3] + "..."
    shown = "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in value
    )
    return r"\text{" + "".join(_TEXT_ESCAPES.get(ch, ch) for ch in shown) + "}"


def name(identifier: str) -> str:
    r"""Return an argument's name as a variable: ``x``, or ``\mathit{burst\_len}``."""
    if len(identifier) == 1 and identifier.isascii() and identifier.isalpha():
        return identifier
    return r"\mathit{" + identifier.replace("_", r"\_") + "}"


def _integer(value: int) -> str:
    magnitude = abs(value)
    if magnitude >= _POWER_FROM:
        sign = "-" if value < 0 else ""
        if magnitude & (magnitude - 1) == 0:
            return f"{sign}2^{{{magnitude.bit_length() - 1}}}"
        # 2**31 - 1; as a negative number it would need parentheses, so it stays digits
        if value > 0 and magnitude & (magnitude + 1) == 0:
            return f"2^{{{magnitude.bit_length()}}} - 1"
    return str(value)


def _float(value: float) -> str:
    if math.isnan(value):
        return r"\text{NaN}"
    if math.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    shown = repr(value)
    if "e" in shown:
        mantissa, exponent = shown.split("e")
        power = f"10^{{{int(exponent)}}}"
        if mantissa in ("1", "1.0"):
            return power
        if mantissa in ("-1", "-1.0"):
            return "-" + power
        return rf"{mantissa} \times {power}"
    return shown


def _is_simple(value: Any) -> bool:
    """Whether ``value`` is written as itself: None, a bool, number, str or enum member."""
    return value is None or type(value) in (bool, int, float, str) or _is_member(value)


def _is_member(value: Any) -> bool:
    return isinstance(value, Enum) and isinstance(value.name, str)


def value(obj: Any) -> str:
    """
    Return a value as LaTeX: a number as digits, a bool, None and an enum member's
    name as text, a string as quoted text, anything else as the text of its repr.
    """
    if obj is None or type(obj) is bool:
        return text(str(obj))
    if type(obj) is int:
        return _integer(obj)
    if type(obj) is float:
        return _float(obj)
    if _is_member(obj):
        return text(obj.name)
    if type(obj) is str:
        return text('"' + obj + '"')
    try:
        shown = repr(obj)
    except Exception:
        shown = f"<{type(obj).__name__}>"
    return text(shown)


def _distinct(values: Sequence[Any]) -> Sequence[Any]:
    """The values without repeats, in order (as they are, when one cannot be hashed)."""
    try:
        return list({(type(item), item): item for item in values}.values())
    except Exception:
        return values


def _set(values: Sequence[Any]) -> str:
    r"""``\{1, 2, 4\}``, each value once, cut past MAX_LISTED; ``\emptyset`` without values."""
    values = _distinct(values)
    if not values:
        return r"\emptyset"
    if len(values) > MAX_LISTED:
        shown = [value(item) for item in values[:_LISTED_HEAD]]
        shown += [r"\dots", value(values[-1])]
    else:
        shown = [value(item) for item in values]
    return r"\{" + ", ".join(shown) + r"\}"


def _interval(low: Any, high: Any) -> str:
    return f"[{value(low)}, {value(high)}]"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def _filtered(var: str, predicate: Any) -> str:
    """The condition a predicate adds, after a comma; nothing without a predicate."""
    if predicate is None:
        return ""
    condition = predicate_latex(predicate, var)
    return r",\ " + (condition if condition is not None else f"p({var})")


def _enum_members(rng_type: RNGEnum[Any]) -> tuple[list[Any], bool]:
    """
    The members an RNGEnum draws from (the weighted ones with a positive weight,
    else all; of those, the ones its predicate accepts), and whether the predicate
    was applied (it is not when it raises).
    """
    weights = rng_type.weights
    if weights:
        candidates = [member for member, weight in weights.items() if weight > 0]
    else:
        candidates = list(rng_type.enum_class.__iter__())
    predicate = rng_type.predicate
    if predicate is None:
        return candidates, True
    try:
        return [member for member in candidates if predicate(member)], True
    except Exception:
        return candidates, False


def _rng_latex(var: str, rng_type: Any) -> str | None:
    """The set a built-in RNG type draws from; None for any other class (a subclass too)."""
    kind = type(rng_type)
    if kind is RNGInteger:
        domain = _interval(rng_type.min, rng_type.max)
        return rf"{var} \in {domain}" + _filtered(var, rng_type.predicate)
    if kind is RNGFloat:
        domain = _interval(rng_type.min, rng_type.max)
        return rf"{var} \in {domain} \subset \mathbb{{R}}" + _filtered(var, rng_type.predicate)
    if kind in (RNGWeightedInteger, RNGWeightedFloat):
        domain = r" \cup ".join(_interval(low, high) for low, high in rng_type.ranges)
        real = r" \subset \mathbb{R}" if kind is RNGWeightedFloat else ""
        return rf"{var} \in {domain}{real}" + _filtered(var, rng_type.predicate)
    if kind is RNGBoolean:
        domain = rf"{var} \in " + _set([True, False])
        probability = rng_type.true_probability
        if probability == 0.5:
            return domain
        return domain + rf",\ P({value(True)}) = {value(probability)}"
    if kind is RNGChoice:
        return rf"{var} \in " + _set(list(rng_type.choices))
    if kind is RNGEnum:
        members, applied = _enum_members(rng_type)
        return rf"{var} \in " + _set(members) + ("" if applied else rf",\ p({var})")
    if kind in (RNGSequence, Series):
        # A predicate was applied when the sequence was built: these are the values left
        return rf"{var} \in " + _set(list(rng_type.sequence))
    if kind is RNGString:
        if rng_type.length is not None:
            length = rf"\lvert {var} \rvert = {value(rng_type.length)}"
        else:
            length = (
                rf"\lvert {var} \rvert \in {_interval(rng_type.min_length, rng_type.max_length)}"
            )
        return length + rf",\ {var} \in \Sigma^{{*}}"
    return None


def argument_latex(arg: Any) -> str | None:
    """
    Return the LaTeX of a test argument (``TestArg.to_dict()``'s ``latex``): its
    fixed value, or the set its built-in RNG type draws from with the condition of
    its predicate. None for a custom RNG type, and when anything fails.
    """
    try:
        var = name(arg.name)
        if arg.is_static:
            return f"{var} = {value(arg._value)}"
        return _rng_latex(var, arg.rng_type)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Functions: finding the source
# ---------------------------------------------------------------------------


def _within(node: ast.AST, code: types.CodeType) -> bool:
    """Whether every instruction of ``code`` that has a position lies inside ``node``."""
    start = (node.lineno, node.col_offset)  # type: ignore[attr-defined]
    end = (node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
    found = False
    for line, end_line, column, end_column in code.co_positions():
        if line is None or end_line is None or column is None or end_column is None:
            continue
        if (line, column) == (end_line, end_column):
            # RESUME, at the first line's column 0, stands for no source
            continue
        if (line, column) < start or (end_line, end_column) > end:
            return False
        found = True
    return found


def _parameters(node: ast.Lambda | ast.FunctionDef) -> list[str] | None:
    """The names of a function's parameters, or None when it has more than plain ones."""
    args = node.args
    if args.vararg or args.kwarg or args.kwonlyargs:
        return None
    return [arg.arg for arg in (*args.posonlyargs, *args.args)]


def _body(node: ast.Lambda | ast.FunctionDef) -> ast.expr | None:
    """The expression a function returns: a lambda's body, or the one ``return`` of a def."""
    if isinstance(node, ast.Lambda):
        return node.body
    statements = list(node.body)
    first = statements[0] if statements else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        # The docstring
        statements = statements[1:]
    if len(statements) == 1 and isinstance(statements[0], ast.Return):
        return statements[0].value
    return None


def _definition(fn: Any) -> tuple[ast.expr, list[str]]:
    """
    Find the source of a plain function: the expression it returns and its
    parameters' names.

    The node is the innermost lambda or def of the function's file that holds
    every instruction of its code (by position, so two lambdas on one line are
    told apart) and has its parameters.

    Raises:
        _Untranslatable: For anything that is not a plain function, without source,
            whose source does not match its code, or with more than one expression
    """
    if not isinstance(fn, types.FunctionType):
        raise _Untranslatable
    code = fn.__code__
    if code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
        raise _Untranslatable
    if code.co_kwonlyargcount or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
        raise _Untranslatable
    filename = code.co_filename
    linecache.checkcache(filename)
    lines = linecache.getlines(filename, fn.__globals__)
    if not lines:
        raise _Untranslatable
    try:
        tree = ast.parse("".join(lines))
    except (SyntaxError, ValueError, RecursionError) as error:
        raise _Untranslatable from error
    expected = list(code.co_varnames[: code.co_argcount])
    best: ast.Lambda | ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Lambda, ast.FunctionDef)):
            continue
        if isinstance(node, ast.FunctionDef) and node.name != code.co_name:
            continue
        if _parameters(node) != expected or not _within(node, code):
            continue
        if best is None or _encloses(best, node):
            best = node
    if best is None:
        raise _Untranslatable
    body = _body(best)
    if body is None:
        raise _Untranslatable
    return body, expected


def _encloses(outer: ast.AST, inner: ast.AST) -> bool:
    """Whether ``inner`` is written inside ``outer``."""
    start = (outer.lineno, outer.col_offset)  # type: ignore[attr-defined]
    end = (outer.end_lineno, outer.end_col_offset)  # type: ignore[attr-defined]
    inner_start = (inner.lineno, inner.col_offset)  # type: ignore[attr-defined]
    inner_end = (inner.end_lineno, inner.end_col_offset)  # type: ignore[attr-defined]
    return start <= inner_start and inner_end <= end


_MISSING = object()


def _bindings(fn: types.FunctionType, parameters: list[str]) -> Callable[[str], Any]:
    """
    Return the lookup of the names a function's expression reads that are not its
    subject: a parameter's default, a closure variable, a global or a builtin.
    """
    defaults = fn.__defaults__ or ()
    bound: dict[str, Any] = dict(zip(parameters[len(parameters) - len(defaults) :], defaults))
    # The first parameter is the subject, whatever its default
    bound.pop(parameters[0], None)
    for free, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        # A cell may still be empty
        with contextlib.suppress(ValueError):
            bound[free] = cell.cell_contents
    local = set(fn.__code__.co_varnames) | set(fn.__code__.co_cellvars)

    def lookup(identifier: str) -> Any:
        if identifier in bound:
            return bound[identifier]
        if identifier in local:
            return _MISSING
        if identifier in fn.__globals__:
            return fn.__globals__[identifier]
        builtins = fn.__globals__.get("__builtins__", {})
        if isinstance(builtins, types.ModuleType):
            builtins = vars(builtins)
        return builtins.get(identifier, _MISSING) if isinstance(builtins, Mapping) else _MISSING

    return lookup


# ---------------------------------------------------------------------------
# Functions: translating the expression
# ---------------------------------------------------------------------------

# Precedences, loosest first. The bit operators share one, below the comparisons,
# so that they are always in parentheses there: few readers know how they bind
_OR, _AND, _NOT, _BITS, _COMPARE, _ADD, _MUL, _UNARY, _POWER, _ATOM = range(1, 11)

_COMPARISONS: dict[type, str] = {
    ast.Eq: "=",
    ast.NotEq: r"\ne",
    ast.Lt: "<",
    ast.LtE: r"\le",
    ast.Gt: ">",
    ast.GtE: r"\ge",
    ast.In: r"\in",
    ast.NotIn: r"\notin",
}
_NEGATED: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
# operator: (LaTeX, precedence); left-associative
_BINARY: dict[type, tuple[str, int]] = {
    ast.Add: ("+", _ADD),
    ast.Sub: ("-", _ADD),
    ast.Mult: (r"\cdot", _MUL),
    ast.Mod: (r"\bmod", _MUL),
    ast.BitOr: (r"\mathbin{|}", _BITS),
    ast.BitXor: (r"\oplus", _BITS),
    ast.BitAnd: (r"\mathbin{\&}", _BITS),
    ast.LShift: (r"\ll", _BITS),
    ast.RShift: (r"\gg", _BITS),
}
# The operators whose result, used as a condition, means "is not zero"
_NUMERIC_TESTS = (ast.Mod, ast.BitAnd)
_FUNCTIONS = {"abs": abs, "len": len, "min": min, "max": max, "range": range}


class _Translator:
    """
    Translates one function's expression.

    Args:
        subject: The LaTeX of the function's first parameter, for a predicate (the
            argument's name); None for a constraint, whose first parameter is the
            row and stands for nothing by itself
        fields: For a constraint, the arguments' names in the row's order
        parameter: The name of the function's first parameter
        lookup: The value bound to any other name (``_MISSING`` when unknown)
    """

    def __init__(
        self,
        *,
        subject: str | None,
        fields: Sequence[str],
        parameter: str,
        lookup: Callable[[str], Any],
    ) -> None:
        self.subject = subject
        self.fields = tuple(fields)
        self.parameter = parameter
        self.lookup = lookup

    # -- conditions ---------------------------------------------------------

    def condition(self, node: ast.expr) -> tuple[str, int]:
        """Translate an expression used as a condition (what the function returns)."""
        if isinstance(node, ast.BoolOp):
            symbol, level = (r"\land", _AND) if isinstance(node.op, ast.And) else (r"\lor", _OR)
            parts = [self._wrap(self.condition(operand), level + 1) for operand in node.values]
            return f" {symbol} ".join(parts), level
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._negation(node.operand)
        if isinstance(node, ast.Compare):
            return self._comparison(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _NUMERIC_TESTS):
            return rf"{self._wrap(self.term(node), _COMPARE + 1)} \ne 0", _COMPARE
        raise _Untranslatable

    def _negation(self, node: ast.expr) -> tuple[str, int]:
        """``not`` of a condition: a single comparison is reversed, the rest gets a ¬."""
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _NEGATED:
            reversed_ = ast.Compare(
                left=node.left, ops=[_NEGATED[type(node.ops[0])]()], comparators=node.comparators
            )
            return self._comparison(reversed_)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _NUMERIC_TESTS):
            return f"{self._wrap(self.term(node), _COMPARE + 1)} = 0", _COMPARE
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.condition(node.operand)
        inner = self.condition(node)
        return rf"\lnot \left({inner[0]}\right)", _NOT

    def _comparison(self, node: ast.Compare) -> tuple[str, int]:
        parts = [self._wrap(self.term(node.left), _COMPARE + 1)]
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)):
                # Identity is written as equality only against None, True and False
                if not (isinstance(right, ast.Constant) and right.value in (None, True, False)):
                    raise _Untranslatable
                if type(right.value) not in (bool, type(None)):
                    raise _Untranslatable
                parts.append("=" if isinstance(op, ast.Is) else r"\ne")
                parts.append(value(right.value))
            elif isinstance(op, (ast.In, ast.NotIn)):
                parts.append(_COMPARISONS[type(op)])
                parts.append(self._collection(right))
            else:
                parts.append(_COMPARISONS[type(op)])
                parts.append(self._wrap(self.term(right), _COMPARE + 1))
        return " ".join(parts), _COMPARE

    # -- collections (the right side of "in") --------------------------------

    def _collection(self, node: ast.expr) -> str:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            if not node.elts or len(node.elts) > MAX_LISTED:
                raise _Untranslatable
            return r"\{" + ", ".join(self.term(item)[0] for item in node.elts) + r"\}"
        if isinstance(node, ast.Call) and self._function(node) is range:
            bounds = [self._constant(argument) for argument in node.args]
            if not bounds or not all(type(bound) is int for bound in bounds):
                raise _Untranslatable
            return self._range(range(*bounds))
        bound = self._known(node)
        if isinstance(bound, range):
            return self._range(bound)
        if isinstance(bound, (tuple, list, set, frozenset)):
            items = list(bound)
            if not items or not all(_is_simple(item) for item in items):
                raise _Untranslatable
            if isinstance(bound, (set, frozenset)):
                # A set's order changes between runs: sorted, or not written
                try:
                    items = sorted(items)
                except TypeError as error:
                    raise _Untranslatable from error
            return _set(items)
        raise _Untranslatable

    @staticmethod
    def _range(numbers: range) -> str:
        """A range of consecutive integers as an interval; any other one is not written."""
        if len(numbers) == 0 or numbers.step not in (1, -1):
            raise _Untranslatable
        return _interval(min(numbers[0], numbers[-1]), max(numbers[0], numbers[-1]))

    # -- terms ---------------------------------------------------------------

    def term(self, node: ast.expr) -> tuple[str, int]:
        """Translate an expression used as a value."""
        if isinstance(node, ast.Constant):
            return self._value(node.value)
        if isinstance(node, ast.Name):
            if node.id == self.parameter:
                if self.subject is None:
                    raise _Untranslatable
                return self.subject, _ATOM
            return self._value(self._known(node))
        if isinstance(node, ast.Attribute):
            field = self._field(node)
            if field is not None:
                return name(field), _ATOM
            return self._value(self._known(node))
        if isinstance(node, ast.Subscript):
            field = self._field(node)
            if field is None:
                raise _Untranslatable
            return name(field), _ATOM
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return "-" + self._wrap(self.term(node.operand), _UNARY), _UNARY
            if isinstance(node.op, ast.UAdd):
                return self.term(node.operand)
            raise _Untranslatable
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise _Untranslatable

    def _binary(self, node: ast.BinOp) -> tuple[str, int]:
        if isinstance(node.op, ast.Div):
            return rf"\frac{{{self.term(node.left)[0]}}}{{{self.term(node.right)[0]}}}", _ATOM
        if isinstance(node.op, ast.FloorDiv):
            left, right = self.term(node.left)[0], self.term(node.right)[0]
            return rf"\left\lfloor \frac{{{left}}}{{{right}}} \right\rfloor", _ATOM
        if isinstance(node.op, ast.Pow):
            base = self._wrap(self.term(node.left), _ATOM)
            return f"{base}^{{{self.term(node.right)[0]}}}", _POWER
        if type(node.op) not in _BINARY:
            raise _Untranslatable
        symbol, level = _BINARY[type(node.op)]
        if level == _BITS:
            # Only a chain of one bit operator goes without parentheses
            left_term = self.term(node.left)
            chained = isinstance(node.left, ast.BinOp) and type(node.left.op) is type(node.op)
            left = left_term[0] if chained else self._wrap(left_term, _POWER)
            right = self._wrap(self.term(node.right), _POWER)
            return f"{left} {symbol} {right}", level
        left = self._wrap(self.term(node.left), level)
        right = self._wrap(self.term(node.right), level + 1)
        if right.startswith("-"):
            right = rf"\left({right}\right)"
        return f"{left} {symbol} {right}", level

    def _call(self, node: ast.Call) -> tuple[str, int]:
        if node.keywords or any(isinstance(argument, ast.Starred) for argument in node.args):
            raise _Untranslatable
        function = self._function(node)
        if function in (abs, len) and len(node.args) == 1:
            return rf"\lvert {self.term(node.args[0])[0]} \rvert", _ATOM
        if function in (min, max) and len(node.args) >= 2:
            arguments = ", ".join(self.term(argument)[0] for argument in node.args)
            return rf"\{function.__name__}\left({arguments}\right)", _ATOM
        raise _Untranslatable

    def _function(self, node: ast.Call) -> Any:
        """The builtin a call names, when the name is still bound to it; else None."""
        if not isinstance(node.func, ast.Name) or node.func.id == self.parameter:
            return None
        expected = _FUNCTIONS.get(node.func.id)
        return expected if expected is not None and self.lookup(node.func.id) is expected else None

    def _field(self, node: ast.Attribute | ast.Subscript) -> str | None:
        """The argument a constraint reads from its row: ``v.addr`` or ``v[0]``."""
        if self.subject is not None:
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id == self.parameter):
            return None
        if isinstance(node, ast.Attribute):
            if node.attr in self.fields:
                return node.attr
            raise _Untranslatable
        key = node.slice
        if isinstance(key, ast.UnaryOp) and isinstance(key.op, ast.USub):
            operand = key.operand
            if isinstance(operand, ast.Constant) and type(operand.value) is int:
                key = ast.Constant(value=-operand.value)
        if (
            isinstance(key, ast.Constant)
            and type(key.value) is int
            and -len(self.fields) <= key.value < len(self.fields)
        ):
            return self.fields[key.value]
        raise _Untranslatable

    # -- bound values ----------------------------------------------------------

    def _known(self, node: ast.expr) -> Any:
        """
        The value of a name that is not the subject, or of an enum member written
        ``Color.RED`` (the only attribute read: reading any other could run code).
        """
        if isinstance(node, ast.Name):
            if node.id == self.parameter:
                raise _Untranslatable
            bound = self.lookup(node.id)
            if bound is _MISSING:
                raise _Untranslatable
            return bound
        if isinstance(node, ast.Attribute):
            owner = self._known(node.value)
            if isinstance(owner, type) and issubclass(owner, Enum):
                member = owner.__members__.get(node.attr)
                if member is not None:
                    return member
        raise _Untranslatable

    def _constant(self, node: ast.expr) -> Any:
        """The value of a literal, a negated literal or a known name."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._constant(node.operand)
            if type(inner) in (int, float):
                return -inner
            raise _Untranslatable
        return self._known(node)

    def _value(self, obj: Any) -> tuple[str, int]:
        if not _is_simple(obj):
            raise _Untranslatable
        shown = value(obj)
        if type(obj) in (int, float):
            if shown.startswith("-"):
                return shown, _UNARY
            if shown.endswith(" - 1"):
                return shown, _ADD
            if r"\times" in shown:
                return shown, _MUL
            if "^" in shown:
                return shown, _POWER
        return shown, _ATOM

    @staticmethod
    def _wrap(rendered: tuple[str, int], level: int) -> str:
        shown, own = rendered
        return shown if own >= level else rf"\left({shown}\right)"


def _translate(fn: Any, *, subject: str | None, fields: Sequence[str]) -> str | None:
    try:
        body, parameters = _definition(fn)
        if not parameters:
            return None
        # Every parameter after the first needs a default: the function is called with one
        if len(fn.__defaults__ or ()) < len(parameters) - 1:
            return None
        translator = _Translator(
            subject=subject,
            fields=fields,
            parameter=parameters[0],
            lookup=_bindings(fn, parameters),
        )
        return translator.condition(body)[0]
    except Exception:
        # _Untranslatable, and anything a strange object does when it is looked at
        return None


def predicate_latex(predicate: Any, subject: str) -> str | None:
    """
    Return the condition of an RNG type's predicate as LaTeX, with ``subject`` (the
    argument's name, as LaTeX) for its parameter; None when it is not translated.
    """
    return _translate(predicate, subject=subject, fields=())


def constraint_latex(constraint: Any, arg_names: Sequence[str]) -> str | None:
    """
    Return the condition of a vector constraint as LaTeX, each ``v.<argument>`` (or
    ``v[<position>]``) as that argument's name; None when it is not translated.
    """
    return _translate(constraint, subject=None, fields=tuple(arg_names))
