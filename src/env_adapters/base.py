from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

class BaseSimEnv(ABC):
    @abstractmethod
    def reset(self) -> Dict[str, Any]: pass
    @abstractmethod
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]: pass
    @abstractmethod
    def _observe(self) -> Dict[str, Any]: pass
    def close(self): pass
