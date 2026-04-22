import concurrent.futures
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class ToolSandbox:
    def __init__(self, timeout: float = 5.0, max_workers: int = 2):
        self.timeout = timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        future = self.executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"⏳ 工具执行超时: {func.__name__}")
            return {"status": "timeout", "error": f"Exceeded {self.timeout}s"}
        except Exception as e:
            logger.error(f"💥 工具执行异常: {func.__name__} -> {e}")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def safe_tool(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
