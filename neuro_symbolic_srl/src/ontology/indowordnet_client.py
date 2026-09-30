"""Thin wrapper around ``pyiwn`` (CFILT IndoWordNet) with Windows-safe I/O.

``pyiwn`` reads its bundled synset/relation dumps with Python's *locale*
default text encoding. On Windows that default is a codepage (e.g. cp1252),
which cannot decode the UTF-8 Devanagari data files and crashes on import.
This module works around that by forcing UTF-8 for the duration of the
``IndoWordNet`` construction call only, so the rest of the process is left
with its normal encoding defaults.
"""
from __future__ import annotations

import builtins
import contextlib
import sys
from typing import Iterator, Optional, Sequence

import pyiwn


class IndoWordNetInitError(RuntimeError):
    """Raised when the IndoWordNet lexical database cannot be loaded."""


@contextlib.contextmanager
def _force_utf8_text_reads() -> Iterator[None]:
    """Temporarily force ``open(...)`` to default to UTF-8 for text mode."""
    original_open = builtins.open

    def _utf8_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_open(file, mode, *args, **kwargs)

    builtins.open = _utf8_open
    try:
        yield
    finally:
        builtins.open = original_open


def _make_stdout_utf8_safe() -> None:
    """Best-effort: avoid crashes if pyiwn ever prints Unicode progress bars."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


_make_stdout_utf8_safe()


class IndoWordNetClient:
    """Loads the Hindi IndoWordNet database and exposes a minimal, typed API.

    Wraps ``pyiwn.IndoWordNet`` so the rest of the codebase depends on this
    module instead of the third-party library's exact surface.
    """

    def __init__(self, language: "pyiwn.Language" = pyiwn.Language.HINDI) -> None:
        self.language = language
        self._iwn = self._load(language)

    @staticmethod
    def _load(language: "pyiwn.Language") -> "pyiwn.IndoWordNet":
        try:
            with _force_utf8_text_reads():
                return pyiwn.IndoWordNet(lang=language)
        except FileNotFoundError as exc:
            raise IndoWordNetInitError(
                "IndoWordNet lexical data was not found. Run "
                "`python -c \"import pyiwn\"` once (or "
                "`scripts/download_resources.sh`) to trigger pyiwn's "
                "one-time download before using IndoWordNetClient."
            ) from exc
        except UnicodeDecodeError as exc:
            raise IndoWordNetInitError(
                "Failed to decode IndoWordNet data files as UTF-8. The local "
                "data dump under pyiwn's data directory may be corrupted; "
                "delete it and re-run the pyiwn download."
            ) from exc

    def synsets(
        self, lemma: str, pos: Optional["pyiwn.PosTag"] = None
    ) -> Sequence["pyiwn.iwn.Synset"]:
        """Return all candidate synsets for ``lemma`` (optionally filtered by POS).

        Returns an empty list for out-of-vocabulary lemmas; pyiwn itself
        raises ``KeyError`` in that case, which we normalize here.
        """
        try:
            return self._iwn.synsets(lemma, pos)
        except KeyError:
            return []

    def hypernyms(self, synset: "pyiwn.iwn.Synset") -> Sequence["pyiwn.iwn.Synset"]:
        """Return the direct hypernym(s) of ``synset`` (empty if none recorded)."""
        return self._iwn.synset_relation(synset, pyiwn.SynsetRelations.HYPERNYMY)

    def gloss(self, synset: "pyiwn.iwn.Synset") -> str:
        return synset.gloss()

    def lemma_names(self, synset: "pyiwn.iwn.Synset") -> Sequence[str]:
        return synset.lemma_names()
