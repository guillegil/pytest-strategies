"""
Unit tests keeping the packaged skill in step with the code (T8.2).

The signature blocks of ``references/api.md`` are the ``python`` blocks whose
functions end in ``...``. Each one is compared with ``inspect.signature`` of the
object it names: the parameter names, their order and kinds, and the defaults.
Annotations are not compared (the code spells them with private aliases). A
dataclass block is compared with ``dataclasses.fields``. Then the skill's content
is checked against the 4.0 traps.
"""

import ast
import dataclasses
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

import pytest_strategy
from pytest_strategy import hookspecs

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "src" / "pytest_strategy" / "skill" / "pytest-strategies"
API_MD = SKILL_DIR / "references" / "api.md"
SKILL_MD = SKILL_DIR / "SKILL.md"

# Names in __all__ that are not code
METADATA = {"__version__", "__author__", "__email__"}

KINDS = {
    "posonlyargs": inspect.Parameter.POSITIONAL_ONLY,
    "args": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "vararg": inspect.Parameter.VAR_POSITIONAL,
    "kwonlyargs": inspect.Parameter.KEYWORD_ONLY,
    "kwarg": inspect.Parameter.VAR_KEYWORD,
}

# The namespace a documented default is evaluated in (_KEEP: Parameter.extend()'s
# "keep the base's setting" default)
DEFAULTS_NAMESPACE: dict[str, Any] = {
    "__builtins__": {},
    "frozenset": frozenset,
    "_KEEP": pytest_strategy.parameters._KEEP,
}


def _python_blocks(text: str) -> list[tuple[int, str]]:
    """
    Return (line, source) for every ```python block of a Markdown text, dedented, the
    indented blocks of list items included.
    """
    return [
        (text[: match.start()].count("\n") + 2, textwrap.dedent(match.group(2)))
        for match in re.finditer(r"^([ \t]*)```python\n(.*?)^\1```", text, re.S | re.M)
    ]


# The methods whose second argument, or values=, is a vector
VECTOR_METHODS = {"add_directed_vector", "add_test_vector"}


def one_element_tuple_vectors(text: str) -> list[str]:
    """
    Return ``line: vector`` for each one-element tuple that a Markdown text's python
    blocks give as a vector: a value of a dict, such as ``directed_vectors={"neg":
    (-2,)}``, or the vector of an ``add_directed_vector()`` or ``add_test_vector()``
    call.
    """
    found = []
    for line, source in _python_blocks(text):
        for node in ast.walk(ast.parse(source)):
            vectors: list[ast.expr] = []
            if isinstance(node, ast.Dict):
                vectors = node.values
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in VECTOR_METHODS
            ):
                vectors = [*node.args[1:2], *(k.value for k in node.keywords if k.arg == "values")]
            found += [
                f"{line + vector.lineno - 1}: {ast.unparse(vector)}"
                for vector in vectors
                if isinstance(vector, ast.Tuple) and len(vector.elts) == 1
            ]
    return found


def _section(text: str, heading: str) -> str:
    """Return the body of the Markdown section whose heading matches ``heading``."""
    match = re.search(rf"^(#+) {heading}[^\n]*\n", text, re.M)
    assert match, f"no section {heading!r}"
    level = len(match.group(1))
    end = re.compile(rf"^#{{1,{level}}} ", re.M).search(text, match.end())
    return text[match.end() : end.start() if end else len(text)]


def _is_stub(node: ast.FunctionDef) -> bool:
    """Whether a function's body is ``...``, after an optional docstring."""
    body = node.body
    if (
        len(body) == 2
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )


def _decorators(node: ast.FunctionDef | ast.ClassDef) -> dict[str, ast.expr]:
    """Map each decorator's name (``dataclass``, ``staticmethod``) to its node."""
    names = {}
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names[target.id] = decorator
        elif isinstance(target, ast.Attribute):
            names[target.attr] = decorator
    return names


def _evaluate(node: ast.expr) -> Any:
    return eval(compile(ast.Expression(node), "<api.md>", "eval"), DEFAULTS_NAMESPACE)


def _same_value(documented: Any, real: Any) -> bool:
    return type(documented) is type(real) and documented == real


def _documented_parameters(node: ast.FunctionDef) -> list[tuple[str, Any, ast.expr | None]]:
    """Return (name, kind, default node) for each parameter of a stub, in order."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += args.defaults
    found: list[tuple[str, Any, ast.expr | None]] = []
    for arg, default in zip(positional, defaults):
        kind = KINDS["posonlyargs"] if arg in args.posonlyargs else KINDS["args"]
        found.append((arg.arg, kind, default))
    if args.vararg:
        found.append((args.vararg.arg, KINDS["vararg"], None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        found.append((arg.arg, KINDS["kwonlyargs"], default))
    if args.kwarg:
        found.append((args.kwarg.arg, KINDS["kwarg"], None))
    return found


def _compare_signature(where: str, node: ast.FunctionDef, real: Any) -> list[str]:
    """Compare a stub with ``inspect.signature(real)``; return the differences."""
    try:
        signature = inspect.signature(real)
    except (TypeError, ValueError) as e:
        return [f"{where}: no signature to compare ({e})"]
    # Parameters starting with "_" are internal and left out of the reference
    actual = [p for p in signature.parameters.values() if not p.name.startswith("_")]
    documented = _documented_parameters(node)
    problems = []
    names = [name for name, _, _ in documented]
    if names != [p.name for p in actual]:
        problems.append(
            f"{where}: documents ({', '.join(names)}), "
            f"the code takes ({', '.join(p.name for p in actual)})"
        )
        return problems
    for (name, kind, default), parameter in zip(documented, actual):
        if kind != parameter.kind:
            problems.append(
                f"{where}: '{name}' is documented {kind.description}, "
                f"the code has {parameter.kind.description}"
            )
        if default is None:
            if parameter.default is not inspect.Parameter.empty:
                problems.append(
                    f"{where}: '{name}' has the default {parameter.default!r}, " "not documented"
                )
        elif parameter.default is inspect.Parameter.empty:
            problems.append(f"{where}: '{name}' is documented with a default it lacks")
        elif not _same_value(_evaluate(default), parameter.default):
            problems.append(
                f"{where}: '{name}' is documented = {ast.unparse(default)}, "
                f"the code's default is {parameter.default!r}"
            )
    return problems


def _compare_method(where: str, node: ast.FunctionDef, cls: type) -> list[str]:
    """Compare a stub method with the attribute of ``cls`` it documents."""
    try:
        attribute = inspect.getattr_static(cls, node.name)
    except AttributeError:
        return [f"{where}: {cls.__name__} has no attribute {node.name!r}"]
    decorators = set(_decorators(node))
    if isinstance(attribute, property):
        if "property" not in decorators:
            return [f"{where}: {node.name} is a property, document it with @property"]
        return []
    if "property" in decorators:
        return [f"{where}: {node.name} is not a property"]
    for wrapper in (staticmethod, classmethod):
        if isinstance(attribute, wrapper) != (wrapper.__name__ in decorators):
            return [f"{where}: {node.name} is decorated differently from the code"]
    function = getattr(attribute, "__func__", attribute)
    return _compare_signature(where, node, function)


def _compare_fields(where: str, node: ast.ClassDef, cls: type) -> list[str]:
    """Compare a dataclass block's fields and options with the dataclass."""
    if not dataclasses.is_dataclass(cls):
        return [f"{where}: {cls.__name__} is not a dataclass"]
    problems = []
    documented = [
        statement
        for statement in node.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    ]
    fields = dataclasses.fields(cls)
    names = [statement.target.id for statement in documented]  # type: ignore[union-attr]
    if names != [field.name for field in fields]:
        return [
            f"{where}: documents the fields ({', '.join(names)}), "
            f"the code has ({', '.join(field.name for field in fields)})"
        ]
    for statement, field in zip(documented, fields):
        if field.default is not dataclasses.MISSING:
            real = field.default
        elif field.default_factory is not dataclasses.MISSING:
            real = field.default_factory()
        else:
            real = dataclasses.MISSING
        if statement.value is None:
            if real is not dataclasses.MISSING:
                problems.append(f"{where}: field '{field.name}' has a default, not documented")
        elif real is dataclasses.MISSING:
            problems.append(f"{where}: field '{field.name}' has no default")
        elif not _same_value(_evaluate(statement.value), real):
            problems.append(
                f"{where}: field '{field.name}' is documented = "
                f"{ast.unparse(statement.value)}, the code's default is {real!r}"
            )
    options = _decorators(node).get("dataclass")
    keywords = {}
    if isinstance(options, ast.Call):
        keywords = {keyword.arg: _evaluate(keyword.value) for keyword in options.keywords}
    params = cls.__dataclass_params__  # type: ignore[attr-defined]
    actual = {
        "frozen": params.frozen,
        "slots": "__slots__" in vars(cls),
        "kw_only": all(field.kw_only for field in fields),
    }
    for option, value in actual.items():
        if keywords.get(option, False) != value:
            problems.append(f"{where}: the dataclass has {option}={value}")
    return problems


def _module_object(name: str) -> Any:
    if name in pytest_strategy.__all__:
        return getattr(pytest_strategy, name)
    if name.startswith("pytest_") and hasattr(hookspecs, name):
        return getattr(hookspecs, name)
    return None


def check_signature_blocks(text: str) -> tuple[list[str], set[str]]:
    """
    Compare every signature block of a Markdown text with the code.

    The signatures are the undecorated top-level functions whose body is ``...``,
    and the classes holding such methods or named in ``__all__`` as dataclasses.
    Return the problems found and the names documented: ``register``,
    ``Parameter.__init__``, ``VectorInfo.seed`` (a field).
    """
    problems: list[str] = []
    documented: set[str] = set()
    for line, source in _python_blocks(text):
        tree = ast.parse(source)
        for node in tree.body:
            where = f"api.md:{line + node.lineno - 1}"
            # A decorated function is an example (a test), not a signature
            if isinstance(node, ast.FunctionDef) and _is_stub(node) and not node.decorator_list:
                real = _module_object(node.name)
                if real is None:
                    problems.append(f"{where}: {node.name} is not a public function")
                    continue
                documented.add(node.name)
                problems += _compare_signature(f"{where} {node.name}", node, real)
            elif isinstance(node, ast.ClassDef):
                stubs = [
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and _is_stub(item)
                ]
                is_dataclass = "dataclass" in _decorators(node)
                if not stubs and (node.name not in pytest_strategy.__all__ or not is_dataclass):
                    continue  # an example class, such as a record type
                cls = _module_object(node.name)
                if not isinstance(cls, type):
                    problems.append(f"{where}: {node.name} is not a public class")
                    continue
                for base in node.bases:
                    base_name = ast.unparse(base).split("[")[0]
                    if base_name not in {klass.__name__ for klass in cls.__mro__}:
                        problems.append(f"{where}: {node.name} is not a subclass of {base_name}")
                if is_dataclass:
                    problems += _compare_fields(f"{where} {node.name}", node, cls)
                    documented |= {
                        f"{node.name}.{statement.target.id}"
                        for statement in node.body
                        if isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                    }
                for stub in stubs:
                    name = f"{node.name}.{stub.name}"
                    stub_where = f"api.md:{line + stub.lineno - 1} {name}"
                    if name in documented:
                        problems.append(f"{stub_where}: documented twice")
                    documented.add(name)
                    problems += _compare_method(stub_where, stub, cls)
    return problems, documented


def _public_members(cls: type) -> set[str]:
    """The public attributes a class defines, with ``__init__``, as Class.name."""
    fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    members = {
        f"{cls.__name__}.{name}"
        for name in vars(cls)
        if not name.startswith("_") and name not in fields
    }
    if "__init__" in vars(cls) and not dataclasses.is_dataclass(cls):
        members.add(f"{cls.__name__}.__init__")
    return members


@pytest.fixture(scope="module")
def api_md() -> str:
    return API_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def skill_md() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def checked(api_md) -> tuple[list[str], set[str]]:
    return check_signature_blocks(api_md)


# ---------------------------------------------------------------------------
# api.md signature blocks against the code
# ---------------------------------------------------------------------------


class TestApiSignatures:
    def test_signature_blocks_match_the_code(self, checked):
        problems, _ = checked
        assert not problems, "\n".join(problems)

    def test_entry_points_are_documented(self, checked):
        _, documented = checked
        rng_types = [
            "RNGInteger",
            "RNGFloat",
            "RNGBoolean",
            "RNGChoice",
            "RNGEnum",
            "RNGString",
            "RNGWeightedInteger",
            "RNGWeightedFloat",
            "Series",
            "RNGSequence",
        ]
        required = {
            "register",
            "strategy",
            "export_strategies",
            "get_context",
            "pytest_strategies_context",
            "Parameter.__init__",
            "TestArg.__init__",
            *(f"{name}.__init__" for name in rng_types),
            *(
                f"StrategyOptions.{f.name}"
                for f in dataclasses.fields(pytest_strategy.StrategyOptions)
            ),
            *(f"VectorInfo.{f.name}" for f in dataclasses.fields(pytest_strategy.VectorInfo)),
        }
        assert not required - documented

    @pytest.mark.parametrize(
        "cls",
        [
            pytest_strategy.Parameter,
            pytest_strategy.TestArg,
            pytest_strategy.RNG,
            pytest_strategy.RNGType,
            pytest_strategy.StrategyOptions,
            pytest_strategy.VectorInfo,
        ],
        ids=lambda cls: cls.__name__,
    )
    def test_public_members_are_documented(self, checked, cls):
        """A method or property added to the code needs a line in api.md."""
        _, documented = checked
        assert not _public_members(cls) - documented

    def test_imports_block_lists_every_public_name(self, api_md):
        _, source = _python_blocks(_section(api_md, "1. Imports"))[0]
        imports = ast.parse(source).body[0]
        assert isinstance(imports, ast.ImportFrom) and imports.module == "pytest_strategy"
        names = [alias.name for alias in imports.names]
        assert sorted(names) == sorted(set(pytest_strategy.__all__) - METADATA)


class TestSignatureCheck:
    """The check itself finds each kind of drift."""

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ("def register(name: str): ...", []),
            ("def register(name, extra): ...", ["documents (name, extra)"]),
            ("def register(): ...", ["documents ()"]),
            ("def register(name_or_factory): ...", ["the code takes (name)"]),
            ("def strategy(name, validate_signature=True): ...", ["keyword-only"]),
            ("def strategy(name, *, validate_signature=False): ...", ["code's default is True"]),
            ("def strategy(name, *, validate_signature=1): ...", ["code's default is True"]),
            ("def strategy(name, *, validate_signature): ...", ["not documented"]),
            ("def strategy(name='x', *, validate_signature=True): ...", ["lacks"]),
            ("def no_such_function(): ...", ["not a public function"]),
            ("def helper():\n    return 1", []),
            ("@strategy('points')\ndef test_scaled(point): ...", []),
            ("class Parameter:\n    def add_vector(self, name): ...", ["no attribute"]),
            ("class Parameter:\n    def arg_names(self): ...", ["is a property"]),
            ("class Parameter:\n    @property\n    def get_arg(self): ...", ["not a property"]),
            ("class RNG:\n    def get_seed() -> int: ...", ["decorated differently"]),
            ("class RNG:\n    @staticmethod\n    def seed(seed=0): ...", ["default is None"]),
            ("class RNGInteger(RNGChoice):\n    def generate(self): ...", ["not a subclass"]),
            ("class Parametre:\n    def get_arg(self, name): ...", ["not a public class"]),
            (
                "class Parameter:\n    def get_arg(self, name): ...\n"
                "    def get_arg(self, name): ...",
                ["documented twice"],
            ),
            ("@dataclass\nclass Point:\n    x: int", []),
            (
                "@dataclass(frozen=True, slots=True, kw_only=True)\nclass StrategyOptions:\n"
                "    nsamples: int = 10\n    strategy: str",
                ["documents the fields (nsamples, strategy)"],
            ),
            (
                "@dataclass(frozen=True, slots=True)\nclass StrategyOptions:\n"
                "    strategy: str\n    nsamples: int = 5\n    nsamples_source: str = 'default'\n"
                "    mode: str = 'all'\n    vector_name: str | None = None\n"
                "    vector_index: int | None = None\n"
                "    constraints_off: frozenset[str] = frozenset()",
                ["code's default is 10", "kw_only=True"],
            ),
        ],
    )
    def test_finds_drift(self, block, expected):
        problems, _ = check_signature_blocks(f"```python\n{block}\n```\n")
        assert len(problems) == len(expected), problems
        for problem, text in zip(problems, expected):
            assert text in problem

    def test_methods_and_fields_are_recorded(self):
        _, documented = check_signature_blocks(
            "```python\nclass TestArg:\n    def generate(self) -> Any: ...\n```\n"
        )
        assert documented == {"TestArg.generate"}


# ---------------------------------------------------------------------------
# Skill content: the 4.0 traps and the removed APIs
# ---------------------------------------------------------------------------


class TestSkillContent:
    def test_api_md_has_a_removed_in_4_table(self, api_md):
        removed = _section(api_md, "Removed in 4.0")
        assert re.search(r"^\| Removed \| What you see \| Use instead \|$", removed, re.M)
        for api in [
            "(argnames, samples)",
            "directed_values",
            "test_values",
            "has_directed_values",
            "always_include_directed",
            "RNG.set_max_retries",
            "configure()",
            "Strategy.set_config()",
        ]:
            assert api in removed

    def test_skill_points_to_the_removed_table(self, api_md, skill_md):
        upgrading = _section(skill_md, "Upgrading from 3.x")
        number = re.search(r"references/api\.md, section (\d+)", upgrading)
        assert number
        assert re.search(rf"^## {number.group(1)}\. Upgrading from 3\.x", api_md, re.M)

    @pytest.mark.parametrize(
        "topic",
        [
            "-k",  # rows are selected by name
            "generate()",  # draw inside an RNG type's generate()
            "strategies_ctx",
            "get_context",
        ],
    )
    def test_traps_cover_the_4_0_changes(self, skill_md, topic):
        assert topic in _section(skill_md, "Traps")

    @pytest.mark.parametrize("path", [SKILL_MD, API_MD], ids=lambda path: path.name)
    def test_no_3_0_advice_left(self, path):
        """3.0 told agents to use (x,) vectors and that names were not in the IDs."""
        text = path.read_text(encoding="utf-8")
        for stale in [
            r"trailing comma",
            r"not (?:part of|in) (?:its |the )?(?:test )?IDs?\b",
            r"IDs look like `x=1,y=2`",
            r"pytest-strategies 3\.0\.0 API",
        ]:
            assert not re.search(stale, text, re.I), stale

    def test_skill_vectors_use_dicts_for_one_argument(self, skill_md):
        """The skill never shows a one-element tuple as a vector."""
        assert len(_python_blocks(skill_md)) == skill_md.count("```python\n")
        assert one_element_tuple_vectors(skill_md) == []

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ('directed_vectors={"neg": {"n": -2}}', []),
            ('directed_vectors={"neg": (-2,)}', ["4: (-2,)"]),
            ('test_vectors={"pair": (1, 2), "one": (3,)}', ["4: (3,)"]),
            ('param.add_directed_vector("neg", (-2,))', ["4: (-2,)"]),
            ('param.add_test_vector("neg", values=(-2,))', ["4: (-2,)"]),
            ('param.add_test_vector("neg", {"n": -2})', []),
            ('param.add_test_vector("pair", (1, 2))', []),
            ("make((-2,), (-3,))", []),
        ],
    )
    def test_one_element_tuple_vectors_are_found(self, block, expected):
        """The block's code starts on line 4 of the text."""
        assert one_element_tuple_vectors(f"Text.\n\n```python\n{block}\n```\n") == expected

    def test_one_element_tuple_vectors_are_found_in_list_items(self):
        """SKILL.md's list items hold indented blocks."""
        text = (
            "- **A trap.** Text:\n\n"
            "  ```python\n"
            '  @register("neg")\n'
            "  def neg():\n"
            "      return Parameter(\n"
            '          TestArg("n", RNGInteger(-9, 9)),\n'
            '          directed_vectors={"neg": (-2,)},\n'
            "      )\n"
            "  ```\n"
        )
        assert _python_blocks(text) == [
            (
                4,
                '@register("neg")\ndef neg():\n    return Parameter(\n'
                '        TestArg("n", RNGInteger(-9, 9)),\n'
                '        directed_vectors={"neg": (-2,)},\n    )\n',
            )
        ]
        assert one_element_tuple_vectors(text) == ["8: (-2,)"]
