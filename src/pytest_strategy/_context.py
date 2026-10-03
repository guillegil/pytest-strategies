"""
The testbench context of a folder: the ``pytest_strategies_context`` implementations
a folder sees, the order the plugin asks them in, the store that calls each one at
most once per session, and the ``conftest.py`` files a folder would see that pytest
has not loaded (:func:`unloaded_conftests`, for ``export_strategies()``).

pluggy calls a hook's implementations by registration order, last registered first,
and pytest registers a ``conftest.py`` when it loads it: the rootdir's and those of
the folders named on the command line before ``pytest_configure``, the others while
it collects. Called through pluggy, which implementation answers for a folder would
then depend on what the command line named. So the plugin orders the
implementations a folder sees itself (:func:`call_order`), calls each one on its
own, and keeps what it answered for every other folder that consults it
(:class:`ContextStore`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import pytest

from ._fingerprint import fingerprint
from ._registry import _contains
from ._streams import StreamKey, path_part, seed_part
from .rng import _Stream

if TYPE_CHECKING:
    from pluggy import HookImpl

# The hook's name, as pluggy and the hook relays know it
HOOK = "pytest_strategies_context"

# What an implementation may raise that the store keeps for the folders that consult
# it, as a factory error, a skip or a failure of each test module that needs ctx.
# Anything else (KeyboardInterrupt, pytest.exit()) goes up at once.
_KEPT = (Exception, pytest.skip.Exception, pytest.fail.Exception)


def is_conftest(impl: HookImpl) -> bool:
    """
    Whether ``impl`` belongs to a ``conftest.py``: pytest registers one under its
    path, so its plugin name ends with ``conftest.py`` (the rule pytest's fixture
    manager uses).
    """
    return impl.plugin_name.endswith("conftest.py")


def _is_wrapper(impl: HookImpl) -> bool:
    return bool(impl.wrapper or impl.hookwrapper)


def _depth(impl: HookImpl) -> int:
    """The depth of a conftest.py implementation's folder."""
    return len(PurePath(os.path.abspath(impl.plugin_name)).parent.parts)


def call_order(impls: Sequence[HookImpl]) -> list[HookImpl]:
    """
    Return the implementations in the order the plugin asks them.

    ``tryfirst`` implementations first, then those of the ``conftest.py`` files
    from the deepest folder up, then the other plugins' in pluggy's order (last
    registered first), then ``trylast`` implementations; within the ``tryfirst``
    and ``trylast`` groups, the conftest.py files from the deepest folder up come
    first too. Wrappers (``wrapper=True`` or ``hookwrapper=True``) come before all
    of them, in the same order among themselves, as pluggy starts a wrapper before
    the implementations it wraps.

    Args:
        impls: The implementations, in pluggy's order (``get_hookimpls()``), which
            is the reverse of the order pluggy calls them in
    """

    def key(item: tuple[int, HookImpl]) -> tuple[int, int, int, int, int]:
        position, impl = item
        kind = 0 if _is_wrapper(impl) else 1
        tier = 0 if impl.tryfirst else 2 if impl.trylast else 1
        if is_conftest(impl):
            return (kind, tier, 0, -_depth(impl), position)
        return (kind, tier, 1, 0, position)

    return [impl for _, impl in sorted(enumerate(reversed(impls)), key=key)]


def folder_of(path: str | os.PathLike[str]) -> str:
    """Return a path's folder, or the path itself when it is a folder, absolute."""
    absolute = os.path.abspath(path)
    return absolute if os.path.isdir(absolute) else os.path.dirname(absolute)


def _sees(folder: str, impl: HookImpl) -> bool:
    """Whether a conftest.py implementation is in ``folder`` (absolute) or above it."""
    own = os.path.dirname(os.path.abspath(impl.plugin_name))
    if _contains(os.path.normcase(own), os.path.normcase(folder)):
        return True
    # The same folders reached through links
    return _contains(
        os.path.normcase(os.path.realpath(own)), os.path.normcase(os.path.realpath(folder))
    )


def visible_from(config: Any, path: str | os.PathLike[str]) -> list[HookImpl]:
    """
    Return the implementations the folder of ``path`` sees, in pluggy's order: every
    plugin's that is not a ``conftest.py``, and those of the loaded ``conftest.py``
    files in that folder and above it.

    A path is a file, whose folder counts, or a folder. This is what a test there
    sees through its folder-scoped hook relay (``item.ihook``), also for a folder
    pytest has not collected, whose relay (``session.gethookproxy(path)``) would
    drop every conftest.py. Without a plugin manager (a unit test's stand-in
    config), it is empty.
    """
    folder = folder_of(path)
    return [impl for impl in _hookimpls(config) if not is_conftest(impl) or _sees(folder, impl)]


def below(path: str | os.PathLike[str], rootpath: str | os.PathLike[str]) -> tuple[str, ...] | None:
    """
    Return the names of the folders from the rootdir down to ``path`` (a folder),
    as it is spelled or by its real path (a rootdir or a folder reached through a
    link); ``()`` for the rootdir itself, and None for a folder outside it.
    """
    for inner, root in (
        (os.path.abspath(path), os.path.abspath(rootpath)),
        (os.path.realpath(path), os.path.realpath(rootpath)),
    ):
        try:
            relative = os.path.relpath(inner, root)
        except ValueError:
            # Another drive on Windows
            continue
        if relative == os.curdir:
            return ()
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            return PurePath(relative).parts
    return None


def unloaded_conftests(config: Any, path: str | os.PathLike[str]) -> list[str]:
    """
    Return the ``conftest.py`` files that pytest loads before it collects a test in
    the folder of ``path`` (a file, or a folder) and that it has not loaded in this
    session, from the rootdir down: those of the rootdir, of that folder and of
    each folder between them, that pytest considers (none above ``--confcutdir``,
    none under ``--noconftest``) and that no folder the session collected made it
    load. The folder's context, as a test there would get it, may come from one of
    them, so it cannot be known.

    A folder outside the rootdir has none, and so does a config without a plugin
    manager (a unit test's stand-in).
    """
    manager = getattr(config, "pluginmanager", None)
    rootpath = getattr(config, "rootpath", None)
    noconftest = getattr(getattr(config, "option", None), "noconftest", False)
    if manager is None or rootpath is None or noconftest:
        return []
    parts = below(folder_of(path), rootpath)
    if parts is None:
        return []
    loaded = {
        os.path.normcase(os.path.realpath(name))
        for name, plugin in manager.list_name_plugin()
        if plugin is not None and name.endswith("conftest.py")
    }
    # pytest's own rule (private API): no conftest.py in a folder above the confcutdir
    considered = getattr(manager, "_is_in_confcutdir", None)
    missing = []
    folder = Path(rootpath)
    for part in (None, *parts):
        if part is not None:
            folder = folder / part
        conftest = folder / "conftest.py"
        if (
            (considered is None or considered(folder))
            and conftest.is_file()
            and os.path.normcase(os.path.realpath(conftest)) not in loaded
        ):
            missing.append(str(conftest))
    return missing


def _hookimpls(config: Any) -> list[HookImpl]:
    """
    Return every implementation of the hook, in pluggy's order, or none without a
    plugin manager (a unit test's stand-in config).
    """
    caller = getattr(getattr(getattr(config, "pluginmanager", None), "hook", None), HOOK, None)
    return [] if caller is None else list(caller.get_hookimpls())


@dataclass(frozen=True, slots=True)
class Answer:
    """
    A folder's context: the value of the implementation that answered (or of the
    implementations a wrapper runs through pluggy), the label that names it in
    messages, or the exception the first implementation asked raised.

    Attributes:
        value: The context, or None when nothing answered
        label: The rootdir-relative path of the conftest.py whose implementation
            answered or raised, the plugin's name for a plugin's, or ``none`` when
            nothing answered. With a wrapper, the deepest conftest.py the folder
            sees that implements the hook.
        error: What the implementation raised, or None
        traceback: The error's traceback when it was raised, so raising it again
            for another folder does not stack frames
        fingerprint: The value's fingerprint (``_fingerprint.fingerprint``), computed
            when the implementation returned it, before anything received it; None
            when the value is None or the implementation raised
        partial: The types in the value that its fingerprint has by name alone
    """

    value: Any
    label: str
    error: BaseException | None = None
    traceback: TracebackType | None = None
    fingerprint: str | None = None
    partial: tuple[str, ...] = ()

    def get(self) -> Any:
        """Return the context, or raise what the implementation raised."""
        if self.error is not None:
            raise self.error.with_traceback(self.traceback)
        return self.value


# The answer of a folder that no implementation answered for
NO_ANSWER = Answer(None, "none")


def _answered(value: Any, label: str, rootpath: Any) -> Answer:
    """
    Return the answer of an implementation (or a wrapper's call) that returned
    ``value``, with its fingerprint: computed now, so a factory, a fixture or a test
    that changes the object later changes no fingerprint.
    """
    if value is None:
        return Answer(None, label)
    digest, partial = fingerprint(value, rootpath)
    return Answer(value, label, fingerprint=digest, partial=partial)


class _Kept:
    """
    An implementation as pluggy's call loop sees it when a wrapper runs: its
    function returns the store's answer for the implementation, so a wrapper does
    not make an implementation run twice.
    """

    __slots__ = (
        "function",
        "argnames",
        "kwargnames",
        "plugin",
        "opts",
        "plugin_name",
        "wrapper",
        "hookwrapper",
        "optionalhook",
        "tryfirst",
        "trylast",
    )

    def __init__(self, store: ContextStore, impl: HookImpl, seed: int | str) -> None:
        for name in self.__slots__[1:]:
            setattr(self, name, getattr(impl, name))
        self.function: Callable[..., Any] = lambda *_: store._call(impl, seed).get()


class ContextStore:
    """
    The contexts of one session: what each ``pytest_strategies_context``
    implementation answered, or the exception it raised (with its traceback),
    computed when a folder first consults it and kept for the session.

    A folder consults the implementations it sees in :func:`call_order` and gets
    the first value that is not None, so the nearest ``conftest.py`` that answers
    wins, one that returns None defers to the one above, and a plugin answers only
    where no conftest.py does. Folders that end at the same implementation share
    its one call and its object. An exception is what a folder gets when the
    implementation that raised it comes before any answer, so it affects only those
    folders.

    When a wrapper is among the implementations, they run through pluggy's call
    loop, so the wrapper can change the answer. Each implementation still runs at
    most once. A folder's answer is kept for its list of implementations, so the
    other tests of the folder, and of folders that see the same ones, look it up.

    Each call, a wrapper's included, draws from the random stream root(S, "ctx")
    (streams v1), started anew for each, so what an implementation draws does not
    depend on which folder asked first or what it asked before.

    The fingerprint of each value is computed right after the call that returned
    it, on the same stream, and kept with it (:class:`Answer`).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._answers: dict[HookImpl, Answer] = {}
        self._folders: dict[tuple[HookImpl, ...], Answer] = {}

    def answer(self, impls: Sequence[HookImpl], seed: int | str) -> Answer:
        """
        Return the context of a folder that sees ``impls`` (in pluggy's order), for
        the run seed ``seed``, calling the implementations it needs that no folder
        consulted yet.
        """
        key = tuple(impls)
        answer = self._folders.get(key)
        if answer is None:
            answer = self._first_answer(call_order(impls), seed)
            self._folders[key] = answer
        return answer

    def _first_answer(self, ordered: list[HookImpl], seed: int | str) -> Answer:
        """Return the first answer of the implementations ``ordered`` (call order)."""
        if any(_is_wrapper(impl) for impl in ordered):
            return self._through_pluggy(ordered, seed)
        for impl in ordered:
            answer = self._call(impl, seed)
            if answer.error is not None or answer.value is not None:
                return answer
        return NO_ANSWER

    def scopes(self) -> dict[str, Answer]:
        """
        Return the answers the folders that consulted the store got, by label,
        sorted by label: one per answering implementation (or per conftest.py that
        ends a list with a wrapper), ``none`` for the folders where nothing
        answered, and the errors.
        """
        found: dict[str, Answer] = {}
        for answer in self._folders.values():
            found.setdefault(answer.label, answer)
        return dict(sorted(found.items()))

    def label(self, impl: HookImpl) -> str:
        """
        Return the label of an implementation: its conftest.py's path relative to
        the rootdir, in posix form, or its plugin's name (its type's qualified name
        for a plugin registered without a name, whose name pluggy makes from its
        ``id()``).
        """
        if is_conftest(impl):
            return path_part(impl.plugin_name, getattr(self.config, "rootpath", None))
        if impl.plugin_name == str(id(impl.plugin)):
            return type(impl.plugin).__qualname__
        return impl.plugin_name

    def elsewhere(self, impls: Sequence[HookImpl]) -> list[str]:
        """
        Return the labels of the loaded conftest.py files that implement the hook
        and that a folder seeing ``impls`` does not see, sorted.
        """
        return sorted(
            {
                self.label(impl)
                for impl in _hookimpls(self.config)
                if is_conftest(impl) and all(impl is not seen for seen in impls)
            }
        )

    def _call(self, impl: HookImpl, seed: int | str) -> Answer:
        """
        Return what one implementation answered, calling it the first time, as
        pluggy does: with the arguments it declares, on its own random stream.
        """
        answer = self._answers.get(impl)
        if answer is None:
            label = self.label(impl)
            try:
                args = [{"config": self.config}[name] for name in impl.argnames]
                with _Stream(lambda: StreamKey.root(seed, "ctx")):
                    value = impl.function(*args)
                    # On the stream too: a repr that draws moves no other stream
                    answer = _answered(value, label, getattr(self.config, "rootpath", None))
            except _KEPT as e:
                answer = Answer(None, label, e, e.__traceback__)
            self._answers[impl] = answer
        return answer

    def _through_pluggy(self, ordered: list[HookImpl], seed: int | str) -> Answer:
        """
        Return the context of a folder whose implementations include a wrapper,
        running them through pluggy's own call loop (``_hookexec``, private API),
        which calls the last one in its list first.
        """
        conftests = [impl for impl in ordered if is_conftest(impl)]
        label = self.label(max(conftests, key=_depth) if conftests else ordered[0])
        methods = [
            impl if _is_wrapper(impl) else _Kept(self, impl, seed) for impl in reversed(ordered)
        ]
        try:
            with _Stream(lambda: StreamKey.root(seed, "ctx")):
                value = self.config.pluginmanager._hookexec(
                    HOOK, cast("list[HookImpl]", methods), {"config": self.config}, True
                )
                if value is None:
                    return NO_ANSWER
                return _answered(value, label, getattr(self.config, "rootpath", None))
        except _KEPT as e:
            return Answer(None, label, e, e.__traceback__)


class FolderContext:
    """
    The context of one folder, computed on first use: what a factory resolved for a
    test there receives as ``ctx``.

    It is created for every resolved strategy and costs nothing until a factory that
    declares ``ctx`` asks for it, so the hook stays lazy.
    """

    __slots__ = ("_store", "_impls", "_folder", "_seed", "_answer", "_seen")

    def __init__(
        self,
        store: ContextStore | None,
        impls: Callable[[], Sequence[HookImpl]],
        folder: Callable[[], str | os.PathLike[str] | None],
        seed: Callable[[], object],
    ) -> None:
        """
        Args:
            store: The session's store, or None without a session (no context)
            impls: Returns the implementations the folder sees, in pluggy's order
            folder: Returns the folder, for the message that says why ctx is None
            seed: Returns the run seed
        """
        self._store = store
        self._impls = impls
        self._folder = folder
        self._seed = seed
        self._answer: Answer | None = None
        self._seen: Sequence[HookImpl] = ()

    def answer(self) -> Answer:
        """Return the folder's answer (computed once)."""
        if self._answer is None:
            if self._store is None:
                self._answer = NO_ANSWER
            else:
                self._seen = self._impls()
                self._answer = self._store.answer(self._seen, seed_part(self._seed()))
        return self._answer

    def __call__(self) -> Any:
        """Return the context, or raise what the implementation that answered raised."""
        return self.answer().get()

    def why_none(self) -> str | None:
        """
        Say why the folder's context is None when a loaded conftest.py elsewhere
        implements the hook, for the error of a factory that failed with ctx None;
        None otherwise.
        """
        if self._store is None or self.answer() is not NO_ANSWER:
            return None
        elsewhere = self._store.elsewhere(self._seen)
        if not elsewhere:
            return None
        where = self._folder()
        folder = (
            path_part(where, getattr(self._store.config, "rootpath", None))
            if where is not None
            else "this folder"
        )
        return (
            f"ctx is None for {folder}: no {HOOK} implementation in this folder or above "
            f"answered (implemented in {', '.join(elsewhere)}; move it to a common parent "
            "conftest)"
        )
