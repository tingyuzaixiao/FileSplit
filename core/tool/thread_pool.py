import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager

from config.logging_config import logger
from core.tool.atomic_counter import AtomicCounter
from core.tool.queue_full_error import QueueFullError


class ThreadPool:
    """
    线上级线程池：异常隔离 + 优雅关闭 + 指标监控 + 资源保护
    适用场景：模型推理、DB查询、外部API调用等阻塞操作
    """

    def __init__(
            self,
            max_workers: int,
            queue_size: int = 100,
            thread_name_prefix: str = "Pool-",
            task_timeout: Optional[float] = None,
            shutdown_timeout: float = 30.0
    ):
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")

        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_timeout = task_timeout
        self.shutdown_timeout = shutdown_timeout
        self._shutdown = False
        self._lock = threading.Lock()

        # 指标统计（线程安全）
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "rejected": 0
        }

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )

        self._task_counter = AtomicCounter(initial_value=queue_size)
        logger.info(
            f"ProductionThreadPool initialized | workers={max_workers} | "
            f"queue_size={queue_size}"
        )

    def _wrap_task(self, fn: Callable, *args, **kwargs) -> Any:
        """任务包装：统一异常捕获 + 超时控制 + 指标更新"""
        start_time = time.time()
        try:
            if self.task_timeout:
                # 使用wait实现任务级超时（避免Future.result阻塞线程）
                future = self._executor.submit(fn, *args, **kwargs)
                done, _ = wait([future], timeout=self.task_timeout)
                if not done:
                    with self._lock:
                        self._stats["timeout"] += 1
                    raise TimeoutError(f"Task exceeded {self.task_timeout}s timeout")
                return future.result()
            else:
                return fn(*args, **kwargs)
        except Exception as e:
            with self._lock:
                self._stats["failed"] += 1
            logger.exception(
                f"Task failed | type={type(e).__name__} | duration={time.time() - start_time:.3f}s"
            )
            raise
        finally:
            with self._lock:
                self._stats["completed"] += 1

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        if self._shutdown:
            raise RuntimeError("ThreadPool is shutting down, cannot submit new tasks")

        with self._lock:
            self._stats["submitted"] += 1

        if self._task_counter.decrement() < 0:
            self._task_counter.increment()
            with self._lock:
                self._stats["rejected"] += 1
            raise QueueFullError(
                f"Task rejected: thread pool queue full (max={self.queue_size}). "
                f"Current stats: {self.get_stats()}"
            ) from None
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, force: bool = False) -> Dict[str, int]:
        """
        优雅关闭线程池
        :param force: True=立即终止（不推荐），False=等待任务完成（带超时）
        :return: 最终统计指标
        """
        if self._shutdown:
            return self.get_stats()

        self._shutdown = True
        logger.info(f"Initiating thread pool shutdown (force={force}) | {self.get_stats()}")

        if force:
            self._executor.shutdown(wait=False)
        else:
            # 先停止接收新任务
            self._executor.shutdown(wait=False)

            # 等待活跃任务完成（带超时保护）
            start = time.time()
            while time.time() - start < self.shutdown_timeout:
                if self._stats["submitted"] <= self._stats["completed"] + self._stats["failed"]:
                    break
                time.sleep(0.5)
            else:
                logger.warning(
                    f"Shutdown timeout ({self.shutdown_timeout}s) reached. "
                    f"Remaining tasks: {self._stats['submitted'] - self._stats['completed'] - self._stats['failed']}"
                )

        final_stats = self.get_stats()
        logger.info(f"ThreadPool shutdown complete | {final_stats}")
        return final_stats

    def get_stats(self) -> Dict[str, int]:
        """获取实时统计指标（线程安全）"""
        with self._lock:
            return self._stats.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(force=False)
        return False

    @contextmanager
    def task_scope(self, task_name: str = "unnamed"):
        """
        任务作用域：自动记录任务耗时 + 异常标注
        用法: with pool.task_scope("model_inference"): model.encode(text)
        """
        start = time.time()
        try:
            yield
        except Exception as e:
            logger.exception(f"Task '{task_name}' failed after {time.time() - start:.3f}s")
            raise
        else:
            logger.debug(f"Task '{task_name}' completed in {time.time() - start:.3f}s")