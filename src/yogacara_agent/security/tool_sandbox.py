import concurrent.futures
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class ToolSandbox:
    """在超时受控的线程池中执行工具调用，并可按白名单限制可用工具。

    注意：线程池并非强隔离沙箱——超时后任务无法被强杀，只能取消尚未
    开始的执行。对完全不受信任的代码应使用进程级隔离。
    """

    def __init__(
        self,
        timeout: float = 5.0,
        max_workers: int = 2,
        allowed_tools: set[str] | None = None,
    ):
        self.timeout = timeout
        self.allowed_tools = allowed_tools
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        name = getattr(func, "__name__", repr(func))
        if self.allowed_tools is not None and name not in self.allowed_tools:
            logger.warning(f"🚫 工具不在白名单: {name}")
            return {"status": "forbidden", "error": f"Tool '{name}' is not allowed"}

        future = self.executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()  # 只能取消排队中的任务；运行中的线程无法强杀
            logger.warning(f"⏳ 工具执行超时: {name}")
            return {"status": "timeout", "error": f"Exceeded {self.timeout}s"}
        except Exception:
            # 异常细节只进日志，不回传给调用方，避免内部实现泄露
            logger.exception(f"💥 工具执行异常: {name}")
            return {"status": "error", "error": "tool execution failed"}

    @staticmethod
    def safe_tool(func: Callable) -> Callable:
        """将工具函数包装为异常安全的可调用对象。"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(f"💥 safe_tool 捕获异常: {getattr(func, '__name__', func)}")
                return {"status": "error", "error": "tool execution failed"}

        return wrapper
