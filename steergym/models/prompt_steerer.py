from dataclasses import dataclass

from ..concept import Concept
from ..steer import SteerConfig, SteerContext, SteerModel


@dataclass
class PromptSteerConfig(SteerConfig):
    # where Concept.description goes relative to the user prompt:
    # "append" (ifeval rule idiom), "prepend", "system" (system-prompt text
    # folded into the user turn — gemma has no system role; personavec idiom)
    placement: str = "append"


class PromptSteerContext(SteerContext):
    def __init__(self, description: str, placement: str):
        self.description = description
        self.placement = placement

    def render(self, prompt: str) -> str:
        if self.placement == "append":
            return f"{prompt}\n{self.description}"
        if self.placement == "prepend":
            return f"{self.description}\n{prompt}"
        if self.placement == "system":
            return f"{self.description}\n\n{prompt}"
        raise ValueError(f"unknown placement {self.placement!r}")


class PromptSteerModel(SteerModel):
    config_cls = PromptSteerConfig

    def context(self, concept: Concept, **coeffs) -> PromptSteerContext:
        if coeffs:
            raise ValueError(f"PromptSteerModel takes no coefficients, got {coeffs}")
        return PromptSteerContext(concept.description, self.config.placement)
