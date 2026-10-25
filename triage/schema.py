"""The decision shape every model reply must pass: call, purpose, evidence timestamps."""

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Call = Literal["keep", "age_restrict", "remove"]


class Evidence(BaseModel):
    start: float = Field(ge=0)
    end: float
    source: Literal["transcript", "frames"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def start_not_after_end(self):
        # start == end is a single instant, e.g. one captioned frame.
        if self.start > self.end:
            raise ValueError(f"evidence start {self.start} is after end {self.end}")
        return self


class Decision(BaseModel):
    call: Call
    purpose: str = Field(min_length=1)
    evidence: list[Evidence]


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_decision(raw: str) -> Decision:
    # Fences are stripped for parsing only; callers keep the raw reply untouched.
    match = _FENCE.match(raw)
    return Decision.model_validate_json(match.group(1) if match else raw)
