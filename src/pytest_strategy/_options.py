"""
The run's options as a strategy factory sees them: ``StrategyOptions``.

The session-wide part (``--nsamples``, ``--vector-mode``, ``--vector-name``,
``--vector-index``, ``--strategy-constraint-off``) is read from the config once
per session, and each strategy gets its own frozen instance, cached by its
resolved name (see ``StrategyRuntime.strategy_options``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import pytest

# The --vector-mode values
VectorMode = Literal["all", "random_only", "directed_only", "mixed", "test"]


@dataclass(frozen=True, kw_only=True, slots=True)
class StrategyOptions:
    """
    The run's options for one strategy, as its factory receives them in ``options``.

    Frozen, and every field is keyword-only, so later releases can add fields
    with defaults without changing what an existing factory sees.

    Attributes:
        strategy: The name the strategy was resolved under, as in the ``-v``
            summary: a registered name, or the qualified name of a factory
            passed by object that no registration names for the test (see
            ``strategy()``)
        nsamples: The ``--nsamples`` value, ``"auto"``, or 10 without the option
        nsamples_source: Where ``nsamples`` came from: ``"--nsamples"`` or ``"default"``
        mode: The ``--vector-mode`` value
        vector_name: The ``--vector-name`` value, or None
        vector_index: The ``--vector-index`` value, or None
        constraints_off: The names of the constraints turned off for this
            strategy: the bare names of ``--strategy-constraint-off`` and the
            names of its ``STRATEGY:NAME`` items aimed at this strategy. They are
            not checked against the strategy's constraints, which do not exist
            until the factory returns.
    """

    strategy: str
    nsamples: int | Literal["auto"] = 10
    nsamples_source: str = "default"
    mode: VectorMode = "all"
    vector_name: str | None = None
    vector_index: int | None = None
    constraints_off: frozenset[str] = frozenset()

    @property
    def filtered(self) -> bool:
        """True when ``--vector-name`` or ``--vector-index`` selects one directed vector."""
        return self.vector_name is not None or self.vector_index is not None


@dataclass(frozen=True)
class SessionOptions:
    """
    The session-wide part of ``StrategyOptions``, read from the config once per session.

    Attributes:
        base: The options every strategy shares; its ``strategy`` is empty
        constraints_off: The ``--strategy-constraint-off`` items, as
            ``(strategy, name)`` pairs whose strategy is None for a bare name
    """

    base: StrategyOptions
    constraints_off: tuple[tuple[str | None, str], ...] = ()

    @property
    def constraints_off_items(self) -> list[str]:
        """The ``--strategy-constraint-off`` items, as written (``S:name`` or ``name``), once each."""
        return list(dict.fromkeys(constraint_off_item(*pair) for pair in self.constraints_off))

    def for_strategy(self, name: str) -> StrategyOptions:
        """Return the options of the strategy resolved under ``name``."""
        return dataclasses.replace(
            self.base,
            strategy=name,
            constraints_off=frozenset(
                off for target, off in self.constraints_off if target is None or target == name
            ),
        )


def parse_constraint_off(value: str) -> tuple[tuple[str | None, str], ...]:
    """
    Split one ``--strategy-constraint-off`` value into ``(strategy, name)`` pairs.

    The value is ``ITEM[,ITEM...]`` with ``ITEM = [STRATEGY:]NAME``. Each item is
    split at its last ``:``, so a strategy name may contain ``:``; the strategy
    is None for a bare name. A strategy whose name contains ``,`` or whitespace
    cannot be targeted.

    Raises:
        ValueError: For whitespace, an empty item, or an empty strategy or name
            (``:x``, ``x:``)
    """
    expected = "Expected ITEM[,ITEM...] with ITEM = [STRATEGY:]NAME"
    if any(c.isspace() for c in value):
        raise ValueError(f"{value!r} contains whitespace. {expected}, without spaces")
    pairs: list[tuple[str | None, str]] = []
    for item in value.split(","):
        if not item:
            raise ValueError(f"{value!r} has an empty item. {expected}")
        strategy, colon, name = item.rpartition(":")
        if not name:
            raise ValueError(f"the item {item!r} has no constraint name. {expected}")
        if colon and not strategy:
            raise ValueError(
                f"the item {item!r} has no strategy name before ':'. {expected}; a bare "
                "NAME turns the constraint off in every strategy"
            )
        pairs.append((strategy if colon else None, name))
    return tuple(pairs)


def constraint_off_item(strategy: str | None, name: str) -> str:
    """
    Write a ``--strategy-constraint-off`` item: ``STRATEGY:NAME``, or ``NAME`` for
    every strategy.

    A strategy name that is empty or contains ``,`` or whitespace cannot be written
    in an item, so for such a strategy the bare ``NAME`` is written, which turns the
    constraint off in every strategy.
    """
    if strategy is None or not strategy or any(c == "," or c.isspace() for c in strategy):
        return name
    return f"{strategy}:{name}"


def _option(config: pytest.Config | None, name: str) -> Any:
    """
    Return the value of the command-line option with dest ``name``, or None.

    A unit test's stand-in config without ``getoption`` (or without this
    option) gives None, so the defaults apply.
    """
    getoption = getattr(config, "getoption", None)
    if getoption is None:
        return None
    return getoption(name, None)


def parse_session_options(config: pytest.Config | None) -> SessionOptions:
    """
    Read the session-wide options from ``config``.

    Without a config, or for an option it does not have, the defaults apply.
    """
    raw_nsamples = _option(config, "nsamples")
    nsamples: int | Literal["auto"]
    if raw_nsamples is None:
        nsamples, source = 10, "default"
    elif raw_nsamples == "auto":
        nsamples, source = "auto", "--nsamples"
    else:
        nsamples, source = int(raw_nsamples), "--nsamples"
    mode = _option(config, "vector_mode")
    # --strategy-constraint-off is repeatable; its values were checked when the
    # command line was parsed (a stand-in may give one str)
    raw_off = _option(config, "strategy_constraint_off")
    values = [raw_off] if isinstance(raw_off, str) else list(raw_off or ())
    return SessionOptions(
        base=StrategyOptions(
            strategy="",
            nsamples=nsamples,
            nsamples_source=source,
            # --vector-mode's choices are the VectorMode values. A stand-in's
            # other value is kept, so generate_vectors() reports it.
            mode=cast(VectorMode, "all" if mode is None else mode),
            vector_name=_option(config, "vector_name"),
            vector_index=_option(config, "vector_index"),
        ),
        constraints_off=tuple(pair for value in values for pair in parse_constraint_off(value)),
    )
