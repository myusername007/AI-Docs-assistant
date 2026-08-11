from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

TOut = TypeVar("TOut", bound=BaseModel)


@dataclass(frozen=True)
class PromptSpec(Generic[TOut]):
    id: str
    version: str
    template: str
    output_model: type[TOut]
    system: str | None = None
    examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self, **variables: object) -> str:
        try:
            return self.template.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Prompt {self.key}: missing variable {e}"
            ) from None