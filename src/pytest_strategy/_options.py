"""
The run's options as a strategy factory sees them: ``StrategyOptions``.

The session-wide part (``--nsamples``, ``--vector-mode``, ``--vector-name``,
``--vector-index``) is read from the config once per session, and each strategy
gets its own frozen instance, cached by its resolved name (see
``StrategyRuntime.strategy_options``).
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
        strategy: The strategy's resolved (registered) name
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

    def for_strategy(self, name: str) -> StrategyOptions:
        """Return the options of the strategy resolved under ``name``."""
        return dataclasses.replace(
            self.base,
            strategy=name,
            constraints_off=frozenset(
                off for target, off in self.constraints_off if target is None or target == name
            ),
        )


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
        )
    )
