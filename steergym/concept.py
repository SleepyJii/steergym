from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    id: int | str
    name: str
    description: str
