"""Pydantic request/response schemas for the SRL parsing API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ParseFrameRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw, space-delimited Hindi sentence to parse.")
    target_predicate: Optional[int] = Field(
        None,
        description=(
            "0-indexed word position of the predicate. If omitted, the "
            "last token is used as a simple SOV-order heuristic (this "
            "service has no dependency parser to auto-detect the verb)."
        ),
    )


class Argument(BaseModel):
    span: List[int] = Field(..., description="[start, end] inclusive 0-indexed word positions.")
    text: str
    karaka: Optional[str] = Field(
        None, description="Paninian Karaka label, if it maps uniquely from the PropBank role."
    )
    role: str = Field(..., description="PropBank role, e.g. ARG0, ARG1, ARGM-INS.")


class Predicate(BaseModel):
    text: str
    lemma: str
    sense: Optional[str] = Field(None, description="IndoWordNet gloss of the disambiguated sense.")
    synset_id: Optional[int] = None


class ParseFrameResponse(BaseModel):
    predicate: Predicate
    arguments: List[Argument]
