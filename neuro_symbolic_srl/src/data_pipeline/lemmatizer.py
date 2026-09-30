"""Lightweight, dictionary-validated Hindi lemma guessers (verb + noun).

Not a full morphological analyzer: each guesser strips one common Hindi
inflection pattern (matras, not independent vowel letters — e.g. the
perfective "ा" in "खोला" is U+093E, not the letter "आ") and checks whether
IndoWordNet actually recognizes the resulting candidate as a headword of
the expected POS, falling back to the surface form otherwise.

This exists because both of ``mask_generator.py``'s core gates are keyed by
dictionary lemma:
- ``arg0_requires_animate`` needs the *predicate's* lemma (e.g. "खोलना") to
  match ``intentional_transitive_verbs``: an inflected surface form (e.g.
  "खोला") would silently never match, disabling the animacy gate entirely.
- ``is_animate``/``is_location``/``is_time`` look the *argument* candidate's
  own lemma up in IndoWordNet: an oblique-case noun (e.g. "बच्चे", the form
  that appears before a postposition) isn't a registered headword either
  ("बच्चा" is), so animacy checks on it silently fail closed.

Both failure modes were caught empirically while building ``api/main.py``,
which has no dependency parser or real lemmatizer for its raw-text
endpoint, and both directly defeated verification-sentence behavior that
the ontology mask is supposed to guarantee.

Known gaps: suppletive/irregular verb perfectives (e.g. जाना -> गया) aren't
recoverable by suffix stripping, and the noun guesser only covers the single
most common oblique pattern (आ-ending masculine singular, e.g. बच्चा/बच्चे) —
other noun classes (feminine, plural, consonant-final) are largely
invariant in the oblique already and don't need it. The CoNLL-U training/
eval path (``hdtb_parser.build_examples_from_conllu``) always uses the
corpus's own gold LEMMA column and never needs either guesser.
"""
from __future__ import annotations

from typing import List, Optional

import pyiwn

from src.ontology.indowordnet_client import IndoWordNetClient

# Longest-first so a greedy strip doesn't cut a shorter suffix out of a
# longer one (e.g. "ाता" before "ा"). All are Devanagari vowel *signs*
# (matras) as they appear word-finally attached to a consonant, not the
# corresponding independent vowel letters.
_VERB_SUFFIXES: List[str] = [
    "ूंगी", "ूंगा", "ेंगी", "ेंगे", "ाती", "ाता", "ाते",
    "ोगी", "ोगे", "ेगा", "ेगी",
    "ती", "ता", "ते", "ीं", "या", "ा", "ी", "े", "ो",
]


def guess_verb_lemma(
    surface_form: str, client: Optional[IndoWordNetClient] = None
) -> str:
    """Best-effort surface -> dictionary-form guess for a Hindi verb.

    Returns ``surface_form`` unchanged if it is already a recognized verb
    headword, or if no suffix-stripped candidate is recognized either.
    """
    client = client or IndoWordNetClient()

    if client.synsets(surface_form, pyiwn.PosTag.VERB):
        return surface_form

    for suffix in _VERB_SUFFIXES:
        if surface_form.endswith(suffix) and len(surface_form) > len(suffix):
            candidate = surface_form[: -len(suffix)] + "ना"
            if client.synsets(candidate, pyiwn.PosTag.VERB):
                return candidate

    return surface_form


_MASCULINE_OBLIQUE_MATRA = "े"  # U+0947
_MASCULINE_DIRECT_MATRA = "ा"  # U+093E


def guess_noun_lemma(
    surface_form: str, client: Optional[IndoWordNetClient] = None
) -> str:
    """Best-effort oblique-case -> direct/dictionary-form guess for a Hindi noun.

    Covers only the single most common pattern — आ-ending masculine
    singular nouns' oblique form (बच्चा -> बच्चे before a postposition) —
    validated against IndoWordNet rather than applied blindly.
    """
    client = client or IndoWordNetClient()

    if client.synsets(surface_form, pyiwn.PosTag.NOUN):
        return surface_form

    if surface_form.endswith(_MASCULINE_OBLIQUE_MATRA) and len(surface_form) > 1:
        candidate = surface_form[:-1] + _MASCULINE_DIRECT_MATRA
        if client.synsets(candidate, pyiwn.PosTag.NOUN):
            return candidate

    return surface_form
