"""动作映射工具。"""

from __future__ import annotations


DEFAULT_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class ActionMapper:
    def __init__(self, actions: list[str] | None = None, mapping: dict[str, int] | None = None):
        self.actions = actions or DEFAULT_ACTIONS
        self.mapping = mapping or {action: idx for idx, action in enumerate(self.actions)}
        self.validate()

    def validate(self) -> None:
        missing = [action for action in self.actions if action not in self.mapping]
        if missing:
            raise ValueError(f"Missing actions in mapping: {missing}")
        values = list(self.mapping.values())
        if len(set(values)) != len(values):
            raise ValueError("Action mapping must be bijective")

    def to_index(self, action: str) -> int:
        if action not in self.mapping:
            raise ValueError(f"Unknown action {action}. Available: {self.actions}")
        return self.mapping[action]

    def to_action(self, index: int) -> str:
        reverse = {idx: action for action, idx in self.mapping.items()}
        if index not in reverse:
            raise ValueError(f"Unknown action index {index}. Available: {sorted(reverse)}")
        return reverse[index]
