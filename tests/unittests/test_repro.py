"""
Unit tests for what a failed strategy row reports (D18): the text of its
``pytest-strategies`` section, the options of its rerun command, how the command
is quoted for POSIX shells and for Windows, the list of failed rows after the
line that says how to reproduce the run, and the properties of a ``--junitxml``
report.
"""

import copy
import shlex
from argparse import Namespace
from pathlib import Path

import pytest
from _pytest._code.code import TerminalRepr

from pytest_strategy import VectorInfo
from pytest_strategy._repro import (
    VALUE_LIMIT,
    SectionedRepr,
    describe,
    generation_options,
    keyword_command,
    outside_rootdir,
    quote,
    rerun_command,
    rerun_nodeid,
    rootdir_nodeid,
    section,
    start_args,
    suite_properties,
    testcase_properties,
)
from pytest_strategy._vector import vector_type
from pytest_strategy.plugin import _failed_rows_lines, _junit_family, _junit_xml

SEED = 1763926297314361000


def info(**fields):
    """A VectorInfo of a random row of ``burst`` (addr=4096, len=17), with ``fields`` changed."""
    defaults = {
        "strategy": "burst",
        "origin": "tests/dma/strategies.py:12",
        "kind": "random",
        "name": None,
        "index": 3,
        "enumerated": (),
        "values": vector_type(("addr", "len"))(4096, 17),
        "id": "rand-3",
        "seed": SEED,
        "context": "3f2a9c1e",
        "constraints_off": (),
    }
    return VectorInfo(**{**defaults, **fields})


class Config:
    """A stand-in config with command-line options."""

    def __init__(self, **options):
        defaults = {
            "nsamples": None,
            "vector_mode": "all",
            "vector_name": None,
            "vector_index": None,
            "strategy_constraint_off": None,
            "override_ini": None,
            "inifilename": None,
            "rootdir": None,
        }
        self.option = Namespace(**{**defaults, **options})

    def getoption(self, name, default=None):
        return getattr(self.option, name, default)


def located(config, root, invocation=None, inipath=None):
    """Give a stand-in ``config`` the rootdir ``root``, its invocation folder and its ini file."""
    config.rootpath = root
    config.inipath = inipath
    config.invocation_params = Namespace(dir=invocation or root)
    return config


def windows_argv(line):
    """
    Split a command line as the Windows C runtime does (CommandLineToArgvW): the
    rule the quotes of ``quote(..., windows=True)`` are written for.
    """
    args, current, in_arg, quoted, i = [], [], False, False, 0
    while i < len(line):
        char = line[i]
        if char == "\\":
            count = len(line[i:]) - len(line[i:].lstrip("\\"))
            if line[i + count : i + count + 1] == '"':
                current.append("\\" * (count // 2))
                if count % 2:
                    current.append('"')
                    i += count + 1
                else:
                    i += count
            else:
                current.append("\\" * count)
                i += count
            in_arg = True
            continue
        if char == '"':
            quoted = not quoted
            in_arg = True
        elif char in " \t" and not quoted:
            if in_arg:
                args.append("".join(current))
                current, in_arg = [], False
        else:
            current.append(char)
            in_arg = True
        i += 1
    if in_arg:
        args.append("".join(current))
    return args


class TestSection:
    def test_the_layout_of_a_random_row(self):
        command = f"pytest 'tests/dma/test_write.py::test_write[rand-3]' --rng-seed={SEED}"

        text = section([info()], command, 0)

        assert text == "\n".join(
            [
                "strategy  burst (tests/dma/strategies.py:12)",
                "vector    rand-3 (random row 3)",
                "values    addr=4096",
                "          len=17",
                f"seed      {SEED}",
                "context   3f2a9c1e",
                f"rerun     {command}",
            ]
        )

    @pytest.mark.parametrize(
        "fields, line",
        [
            (
                {"kind": "directed", "name": "zeros", "index": 0, "id": "directed-zeros"},
                "directed-zeros (directed vector 'zeros', #0)",
            ),
            (
                {"kind": "test", "name": "max", "index": 1, "id": "test-max"},
                "test-max (test vector 'max', #1)",
            ),
            (
                {"kind": "exhaustive", "index": 5, "id": "ch=2", "enumerated": ("addr",)},
                "ch=2 (exhaustive row 5)",
            ),
            ({"kind": "random", "index": 1, "id": "ch=2-rand-1"}, "ch=2-rand-1 (random row 1)"),
            ({"kind": "skipped", "index": None, "id": "skipped"}, "skipped (skipped row)"),
            ({"id": "custom"}, "custom (random row 3)"),
        ],
        ids=["directed", "test", "exhaustive", "enumerated", "skipped", "custom_id"],
    )
    def test_the_vector_line_names_the_row(self, fields, line):
        text = section([info(**fields)], "pytest x", 0)

        assert text.splitlines()[1] == f"vector    {line}"

    def test_no_context_line_without_a_context(self):
        text = section([info(context=None)], "pytest x", 0)

        assert "context" not in text
        assert text.splitlines()[-2:] == [f"seed      {SEED}", "rerun     pytest x"]

    def test_no_origin_when_the_factory_file_is_unknown(self):
        assert section([info(origin=None)], "pytest x", 0).splitlines()[0] == "strategy  burst"

    def test_stacked_strategies_get_a_block_each_and_one_rerun_line(self):
        other = info(
            strategy="mode",
            origin="strategies.py:30",
            kind="directed",
            name="fast",
            index=0,
            id="directed-fast",
            values=vector_type(("mode",))("fast"),
            context=None,
        )

        lines = section([info(), other], "pytest x", 0).splitlines()

        assert lines[6:] == [
            "",
            "strategy  mode (strategies.py:30)",
            "vector    directed-fast (directed vector 'fast', #0)",
            "values    mode='fast'",
            f"seed      {SEED}",
            "rerun     pytest x",
        ]
        assert sum(line.startswith("rerun") for line in lines) == 1

    def test_values_are_stable_reprs(self):
        row = vector_type(("tags", "obj"))(frozenset({"b", "a", "c"}), object())

        lines = section([info(values=row)], "pytest x", 0).splitlines()

        assert lines[2:4] == ["values    tags=frozenset({'a', 'b', 'c'})", "          obj=object"]

    def test_a_value_whose_repr_raises_is_shown_by_its_type(self):
        class Reg:
            def __repr__(self):
                raise RuntimeError("no repr while the device is off")

        class Bank:
            def __repr__(self):
                raise ValueError("no")

        row = vector_type(("reg", "banks", "n"))(Reg(), [Bank()], 2)

        lines = section([info(values=row)], "pytest x", 0).splitlines()

        assert lines[2:5] == [
            "values    reg=<Reg: repr() raised RuntimeError>",
            "          banks=<list: repr() raised ValueError>",
            "          n=2",
        ]

    def test_a_long_value_is_cut_below_vv(self):
        row = vector_type(("blob",))("x" * 5000)

        cut = section([info(values=row)], "pytest x", 1).splitlines()[2]
        full = section([info(values=row)], "pytest x", 2).splitlines()[2]

        shown = "'" + "x" * (VALUE_LIMIT - 1)
        assert cut == f"values    blob={shown}... (5002 characters; -vv shows all)"
        assert full == "values    blob='" + "x" * 5000 + "'"

    def test_a_value_of_exactly_the_limit_is_not_cut(self):
        row = vector_type(("blob",))("x" * (VALUE_LIMIT - 2))

        line = section([info(values=row)], "pytest x", 0).splitlines()[2]

        assert line.endswith("x'")

    def test_the_lines_of_a_multiline_repr_are_aligned(self):
        class Table:
            def __repr__(self):
                return "Table(\n  a=1,\n)"

        row = vector_type(("t", "n"))(Table(), 2)

        lines = section([info(values=row)], "pytest x", 0).splitlines()

        assert lines[2:6] == [
            "values    t=Table(",
            "            a=1,",
            "          )",
            "          n=2",
        ]

    def test_a_note_follows_the_rerun_line(self):
        lines = section([info()], "pytest x", 0, note="outside").splitlines()

        assert lines[-2:] == ["rerun     pytest x", "note      outside"]


class TestSectionedRepr:
    class Lookup(TerminalRepr):
        """A report that takes no sections, as pytest's FixtureLookupErrorRepr."""

        def __init__(self):
            self.argname = "not_a_fixture"

        def toterminal(self, tw):
            tw.line("E       fixture 'not_a_fixture' not found")

    def test_the_report_then_its_sections(self):
        wrapped = SectionedRepr(self.Lookup())

        wrapped.addsection("pytest-strategies", "strategy  burst\nseed      21")

        lines = str(wrapped).splitlines()
        assert lines[0] == "E       fixture 'not_a_fixture' not found"
        assert lines[1].strip("-") == " pytest-strategies "
        assert lines[2:] == ["strategy  burst", "seed      21"]

    def test_the_report_s_other_attributes(self):
        wrapped = SectionedRepr(self.Lookup())

        assert wrapped.argname == "not_a_fixture"
        assert not hasattr(wrapped, "reprcrash")
        # A copy (as pytest-xdist or another plugin may make) works too
        assert copy.copy(wrapped).argname == "not_a_fixture"


class TestDescribe:
    @pytest.mark.parametrize(
        "fields, text",
        [
            ({}, "burst random 3"),
            ({"kind": "directed", "name": "zeros", "index": 0}, "burst directed zeros"),
            ({"kind": "test", "name": "max", "index": 1}, "burst test max"),
            ({"kind": "exhaustive", "index": 5}, "burst exhaustive 5"),
            ({"kind": "skipped", "index": None}, "burst skipped"),
        ],
        ids=["random", "directed", "test", "exhaustive", "skipped"],
    )
    def test_kind_and_row(self, fields, text):
        assert describe([info(**fields)]) == text

    def test_stacked_strategies(self):
        other = info(strategy="mode", kind="directed", name="fast", index=0)

        assert describe([info(), other]) == "burst random 3, mode directed fast"


class TestGenerationOptions:
    def test_none_by_default(self):
        assert generation_options(Config(), [info()]) == []

    def test_every_option_the_run_gave_in_order(self):
        config = Config(
            nsamples=13,
            vector_mode="test",
            vector_name="zeros",
            vector_index=0,
            override_ini=["strategies_ids=values", "python_files=*.py"],
            inifilename="ci/pytest.ini",
            rootdir=".",
            strategy_constraint_off=["aligned"],
        )

        args = generation_options(config, [info(constraints_off=("aligned",))])

        assert args == [
            "--nsamples=13",
            "--vector-mode=test",
            "--vector-name=zeros",
            "--vector-index=0",
            "-o",
            "strategies_ids=values",
            "-c",
            "ci/pytest.ini",
            "--rootdir=.",
            "--strategy-constraint-off=burst:aligned",
        ]

    def test_nsamples_auto(self):
        assert generation_options(Config(nsamples="auto"), [info()]) == ["--nsamples=auto"]

    def test_nsamples_given_at_the_default_value_is_kept(self):
        # --nsamples=10 overrides Parameter(nsamples=...), so it is not the default
        assert generation_options(Config(nsamples=10), [info()]) == ["--nsamples=10"]

    def test_the_last_override_of_an_ini_option_wins(self):
        config = Config(
            override_ini=[
                "strategies_ids=values",
                "strategies_max_exhaustive=50",
                "strategies_ids=names",
            ]
        )

        args = generation_options(config, [info()])

        assert args == ["-o", "strategies_max_exhaustive=50", "-o", "strategies_ids=names"]

    def test_only_the_row_strategies_constraints_turned_off(self):
        # The run turned off a_only and b_only; this row's strategy has only a_only
        config = Config(strategy_constraint_off=["a_only,b_only"])
        row = info(strategy="strat_a", constraints_off=("a_only",))

        assert generation_options(config, [row]) == ["--strategy-constraint-off=strat_a:a_only"]

    def test_the_constraints_of_stacked_strategies_once_each(self):
        rows = [
            info(strategy="a", constraints_off=("x", "y")),
            info(strategy="b", constraints_off=("x",)),
            info(strategy="a", constraints_off=("y",)),
        ]

        args = generation_options(Config(), rows)

        assert args == ["--strategy-constraint-off=a:x,a:y,b:x"]

    def test_a_strategy_name_an_item_cannot_hold_gets_the_bare_name(self):
        row = info(strategy="a,b", constraints_off=("x",))

        assert generation_options(Config(), [row]) == ["--strategy-constraint-off=x"]

    def test_from_another_folder_the_paths_are_relative_to_it(self, tmp_path):
        # pytest -c ci/pytest.ini --rootdir=. started in tmp_path, run from tmp_path/ci
        config = located(
            Config(inifilename="ci/pytest.ini", rootdir="."),
            tmp_path,
            inipath=tmp_path / "ci" / "pytest.ini",
        )

        here = generation_options(config, [info()])
        there = generation_options(config, [info()], start=tmp_path / "ci")

        assert here == ["-c", "ci/pytest.ini", "--rootdir=."]
        assert there == ["-c", "pytest.ini", "--rootdir=.."]

    def test_from_the_invocation_folder_the_paths_are_kept_as_given(self, tmp_path):
        config = located(
            Config(inifilename=str(tmp_path / "ci" / "pytest.ini")),
            tmp_path / "ci",
            invocation=tmp_path,
            inipath=tmp_path / "ci" / "pytest.ini",
        )

        assert generation_options(config, [info()]) == ["-c", str(tmp_path / "ci" / "pytest.ini")]
        assert generation_options(config, [info()], start=tmp_path / "ci") == ["-c", "pytest.ini"]


class TestQuote:
    @pytest.mark.parametrize(
        "arg, quoted",
        [
            ("tests/test_w.py::test_w[rand-3]", "'tests/test_w.py::test_w[rand-3]'"),
            ("--rng-seed=5", "--rng-seed=5"),
            ("it's", "'it'\"'\"'s'"),
            ("", "''"),
        ],
    )
    def test_posix(self, arg, quoted):
        assert quote(arg, windows=False) == quoted
        assert shlex.split(quote(arg, windows=False)) == [arg]

    @pytest.mark.parametrize(
        "arg, quoted",
        [
            ("--rng-seed=5", "--rng-seed=5"),
            ("ci\\pytest.ini", "ci\\pytest.ini"),
            ("tests\\test_w.py::test_w[rand-3]", '"tests\\test_w.py::test_w[rand-3]"'),
            ("--strategy-constraint-off=a:x,b:y", '"--strategy-constraint-off=a:x,b:y"'),
            ("a b", '"a b"'),
            ('say "hi"', '"say \\"hi\\""'),
            ("my dir\\", '"my dir\\\\"'),
            ("", '""'),
        ],
    )
    def test_windows(self, arg, quoted):
        assert quote(arg, windows=True) == quoted

    @pytest.mark.parametrize(
        "arg",
        [
            "tests\\dma\\test_w.py::test_w[ch=2-rand-1]",
            "--vector-name=two words",
            'a "quoted" name',
            'back\\"slash',
            "trailing\\\\",
            "100%",
            "a&b|c<d>e^f",
            "",
        ],
    )
    def test_windows_quotes_survive_the_c_runtime(self, arg):
        line = f"pytest {quote(arg, windows=True)} --rng-seed=5"

        assert windows_argv(line) == ["pytest", arg, "--rng-seed=5"]

    def test_the_platform_rule_by_default(self, monkeypatch):
        monkeypatch.setattr("pytest_strategy._repro.os.name", "nt")
        assert quote("a[1]") == '"a[1]"'
        monkeypatch.setattr("pytest_strategy._repro.os.name", "posix")
        assert quote("a[1]") == "'a[1]'"


class TestRerunNodeid:
    def item(self, tmp_path, path, nodeid, invocation, rootdir):
        config = Namespace(
            rootpath=rootdir,
            invocation_params=Namespace(dir=invocation),
            cwd_relative_nodeid=lambda n: f"cwd:{n}",
        )
        return Namespace(path=path, nodeid=nodeid, config=config)

    def test_inside_the_rootdir_pytest_s_rule(self, tmp_path):
        item = self.item(
            tmp_path,
            tmp_path / "t" / "test_x.py",
            "t/test_x.py::test_a[rand-1]",
            tmp_path,
            tmp_path,
        )

        assert not outside_rootdir(item)
        assert rerun_nodeid(item) == "cwd:t/test_x.py::test_a[rand-1]"

    def test_outside_the_rootdir_the_file_relative_to_the_invocation_folder(self, tmp_path):
        # pytest named it relative to the path given on the command line: "" here
        item = self.item(
            tmp_path,
            tmp_path / "tests" / "test_x.py",
            "::TestA::test_a[rand-1]",
            tmp_path,
            tmp_path / "ci",
        )

        assert outside_rootdir(item)
        assert rerun_nodeid(item) == str(Path("tests", "test_x.py")) + "::TestA::test_a[rand-1]"

    def test_from_the_rootdir_the_node_id_or_the_file_relative_to_it(self, tmp_path):
        inside = self.item(
            tmp_path, tmp_path / "t" / "test_x.py", "t/test_x.py::test_a", tmp_path / "t", tmp_path
        )
        outside = self.item(
            tmp_path, tmp_path / "tests" / "test_x.py", "::test_a", tmp_path, tmp_path / "ci"
        )

        assert rootdir_nodeid(inside) == "t/test_x.py::test_a"
        assert rootdir_nodeid(outside) == str(Path("..", "tests", "test_x.py")) + "::test_a"

    @pytest.mark.parametrize(
        ("args", "start", "expected"),
        [
            (["tests", "tests/test_x.py::test_a"], ".", ["tests", "tests/test_x.py::test_a"]),
            (["./tests/"], ".", ["tests"]),
            (["{root}"], ".", ["."]),
            (["tests"], "ci", ["../tests"]),
            (["{root}"], "ci", [".."]),
        ],
        ids=["as_given", "normalized", "invocation_folder", "from_ci", "from_ci_to_it"],
    )
    def test_the_paths_the_run_started_from(self, tmp_path, args, start, expected):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.py").touch()
        config = Namespace(
            args=[arg.format(root=tmp_path) for arg in args],
            invocation_params=Namespace(dir=tmp_path),
        )

        found = start_args(config, tmp_path / start)

        # In the platform's form
        parts = (text.partition("::") for text in expected)
        assert found == [str(Path(*path.split("/"))) + sep + names for path, sep, names in parts]

    def test_no_paths_when_one_is_not_a_path(self, tmp_path):
        (tmp_path / "tests").mkdir()
        config = Namespace(args=["tests", "acme.tests"], invocation_params=Namespace(dir=tmp_path))

        assert start_args(config, tmp_path) is None

    def test_the_k_command(self):
        command = keyword_command([".", "a b"], 5, ["-c", "ci/pytest.ini"], "m.py and t[x]")

        assert command == " ".join(
            ["pytest", ".", quote("a b"), "--rng-seed=5", "-c", "ci/pytest.ini"]
            + ["-k", quote("m.py and t[x]")]
        )

    def test_the_command(self):
        nodeid = "t.py::test_a[it's]"

        command = rerun_command(nodeid, 5, ["--nsamples=3", "--vector-name=a b"])

        assert command == " ".join(
            ["pytest", quote(nodeid), "--rng-seed=5", "--nsamples=3", quote("--vector-name=a b")]
        )


class TestFailedRowsLines:
    def rows(self, count):
        return {
            f"t.py::test_a[rand-{i}]": {
                "command": f"pytest 't.py::test_a[rand-{i}]' --rng-seed=5",
                "row": f"burst random {i}",
            }
            for i in range(count)
        }

    def test_nothing_without_failed_rows(self):
        assert _failed_rows_lines({}, 0) == []

    def test_each_command_with_its_row(self):
        assert _failed_rows_lines(self.rows(2), 0) == [
            "pytest-strategies: failed rows:",
            "  pytest 't.py::test_a[rand-0]' --rng-seed=5  # burst random 0",
            "  pytest 't.py::test_a[rand-1]' --rng-seed=5  # burst random 1",
        ]

    @pytest.mark.parametrize("verbosity", [0, -1])
    def test_at_most_ten_below_v(self, verbosity):
        lines = _failed_rows_lines(self.rows(13), verbosity)

        assert len(lines) == 12
        assert lines[10] == "  pytest 't.py::test_a[rand-9]' --rng-seed=5  # burst random 9"
        assert lines[-1] == "  ... and 3 more"

    def test_ten_have_no_more_line(self):
        assert _failed_rows_lines(self.rows(10), 0)[-1].endswith("# burst random 9")

    @pytest.mark.parametrize("verbosity", [1, 2])
    def test_all_of_them_with_v(self, verbosity):
        lines = _failed_rows_lines(self.rows(13), verbosity)

        assert len(lines) == 14
        assert lines[-1].endswith("# burst random 12")

    @pytest.mark.parametrize("verbosity", [-2, -3])
    def test_none_under_qq(self, verbosity):
        assert _failed_rows_lines(self.rows(3), verbosity) == []


class TestTestcaseProperties:
    def item(self, tmp_path, nodeid="tests/test_w.py::test_w[rand-3]", **options):
        config = located(Config(**options), tmp_path, invocation=tmp_path / "tests")
        return Namespace(config=config, nodeid=nodeid, path=tmp_path / "tests" / "test_w.py")

    def test_a_random_row(self, tmp_path):
        properties = testcase_properties(self.item(tmp_path), [info()])

        assert properties == [
            ("pytest_strategies.strategy", "burst"),
            ("pytest_strategies.kind", "random"),
            ("pytest_strategies.index", "3"),
            ("pytest_strategies.id", "rand-3"),
            ("pytest_strategies.value.addr", "4096"),
            ("pytest_strategies.value.len", "17"),
            ("pytest_strategies.seed", str(SEED)),
            ("pytest_strategies.context", "3f2a9c1e"),
            (
                "pytest_strategies.command",
                f"pytest {quote('tests/test_w.py::test_w[rand-3]')} --rng-seed={SEED}",
            ),
        ]

    def test_every_value_is_a_string(self, tmp_path):
        row = vector_type(("n", "ok", "none", "tags"))(1, True, None, {"b", "a"})

        properties = testcase_properties(self.item(tmp_path), [info(values=row)])

        assert all(isinstance(value, str) for _, value in properties)
        assert properties[4:8] == [
            ("pytest_strategies.value.n", "1"),
            ("pytest_strategies.value.ok", "True"),
            ("pytest_strategies.value.none", "None"),
            ("pytest_strategies.value.tags", "{'a', 'b'}"),
        ]

    def test_a_value_whose_repr_raises_is_shown_by_its_type(self, tmp_path):
        class Reg:
            def __repr__(self):
                raise RuntimeError("no repr while the device is off")

        row = vector_type(("reg",))(Reg())

        names = dict(testcase_properties(self.item(tmp_path), [info(values=row)]))

        assert names["pytest_strategies.value.reg"] == "<Reg: repr() raised RuntimeError>"

    def test_a_directed_row_has_a_name(self, tmp_path):
        row = info(kind="directed", name="zeros", index=0, id="directed-zeros")

        names = dict(testcase_properties(self.item(tmp_path), [row]))

        assert names["pytest_strategies.name"] == "zeros"
        assert names["pytest_strategies.index"] == "0"

    def test_no_name_index_context_or_constraints_when_unset(self, tmp_path):
        row = info(kind="skipped", index=None, id="skipped", context=None)

        keys = [key for key, _ in testcase_properties(self.item(tmp_path), [row])]

        assert keys == [
            "pytest_strategies.strategy",
            "pytest_strategies.kind",
            "pytest_strategies.id",
            "pytest_strategies.value.addr",
            "pytest_strategies.value.len",
            "pytest_strategies.seed",
            "pytest_strategies.command",
        ]

    def test_the_constraints_turned_off(self, tmp_path):
        row = info(constraints_off=("aligned", "no_4k_cross"))

        names = dict(testcase_properties(self.item(tmp_path), [row]))

        assert names["pytest_strategies.constraints_off"] == "aligned,no_4k_cross"
        assert names["pytest_strategies.command"].endswith(
            quote("--strategy-constraint-off=burst:aligned,burst:no_4k_cross")
        )

    def test_stacked_strategies_are_numbered(self, tmp_path):
        other = info(
            strategy="mode",
            kind="directed",
            name="fast",
            index=0,
            id="directed-fast",
            values=vector_type(("mode",))("fast"),
            context=None,
        )

        properties = testcase_properties(self.item(tmp_path), [info(), other])

        keys = [key for key, _ in properties]
        assert keys[:2] == ["pytest_strategies.0.strategy", "pytest_strategies.0.kind"]
        assert keys[9:] == [
            "pytest_strategies.1.strategy",
            "pytest_strategies.1.kind",
            "pytest_strategies.1.name",
            "pytest_strategies.1.index",
            "pytest_strategies.1.id",
            "pytest_strategies.1.value.mode",
            "pytest_strategies.1.seed",
            "pytest_strategies.1.command",
        ]
        commands = {value for key, value in properties if key.endswith(".command")}
        assert len(commands) == 1

    def test_the_command_runs_from_the_rootdir(self, tmp_path):
        # Started in tmp_path/tests with -c ../ci/pytest.ini --rootdir=..
        item = self.item(tmp_path, inifilename="../ci/pytest.ini", rootdir="..", nsamples=13)
        item.config.inipath = tmp_path / "ci" / "pytest.ini"

        names = dict(testcase_properties(item, [info()]))

        assert names["pytest_strategies.command"] == " ".join(
            [
                "pytest",
                quote("tests/test_w.py::test_w[rand-3]"),
                f"--rng-seed={SEED}",
                "--nsamples=13",
                "-c",
                quote(str(Path("ci", "pytest.ini"))),
                "--rootdir=.",
            ]
        )

    def test_a_long_value_is_cut_below_vv(self, tmp_path):
        row = vector_type(("blob",))("x" * 5000)

        cut = dict(testcase_properties(self.item(tmp_path), [info(values=row)]))
        full = dict(testcase_properties(self.item(tmp_path, verbose=2), [info(values=row)]))

        assert cut["pytest_strategies.value.blob"].endswith("... (5002 characters; -vv shows all)")
        assert full["pytest_strategies.value.blob"] == "'" + "x" * 5000 + "'"


class TestSuiteProperties:
    def test_the_seed_without_failed_rows(self):
        assert suite_properties(SEED, []) == [("pytest_strategies.seed", str(SEED))]

    def test_each_failed_row_s_command_from_0(self):
        rows = [
            {"command": "pytest 't.py::test_a[rand-1]' --rng-seed=5", "row": "burst random 1"},
            {"row": "no command"},
            {"command": "pytest 't.py::test_a[rand-0]' --rng-seed=5", "row": "burst random 0"},
        ]

        assert suite_properties(5, rows) == [
            ("pytest_strategies.seed", "5"),
            ("pytest_strategies.failed.0", "pytest 't.py::test_a[rand-1]' --rng-seed=5"),
            ("pytest_strategies.failed.1", "pytest 't.py::test_a[rand-0]' --rng-seed=5"),
        ]


class TestJunitFamily:
    class Config:
        def __init__(self, xmlpath, family="xunit2"):
            self.option = Namespace(xmlpath=xmlpath)
            self.family = family

        def getini(self, name):
            assert name == "junit_family"
            if self.family is None:
                raise ValueError(f"unknown configuration value: {name!r}")
            return self.family

    @pytest.mark.parametrize(
        "family, expected",
        [("xunit2", "xunit2"), ("xunit1", "xunit1"), ("legacy", "xunit1")],
    )
    def test_the_family_of_the_report(self, family, expected):
        assert _junit_family(self.Config("report.xml", family)) == expected

    def test_none_without_a_report(self):
        assert _junit_family(self.Config(None, "xunit1")) is None
        assert _junit_family(Namespace(option=Namespace())) is None

    def test_none_without_the_junitxml_plugin(self):
        assert _junit_family(self.Config("report.xml", None)) is None


class TestJunitXml:
    def test_the_object_in_pytest_s_stash(self):
        from _pytest.junitxml import xml_key

        config = Namespace(stash=pytest.Stash())
        assert _junit_xml(config) is None
        xml = object()
        config.stash[xml_key] = xml
        assert _junit_xml(config) is xml

    def test_none_when_pytest_has_no_such_key(self, monkeypatch):
        from _pytest.junitxml import xml_key

        config = Namespace(stash=pytest.Stash())
        config.stash[xml_key] = object()
        monkeypatch.delattr("_pytest.junitxml.xml_key")

        assert _junit_xml(config) is None
