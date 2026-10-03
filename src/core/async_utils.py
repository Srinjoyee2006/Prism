"""
Asynchronous concurrency and utility helpers for full-duplex voice pipelines.

Provides cancellation-safe task management, atomic-flushing queues,
and decoupled event dispatching using asyncio primitives.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from typing import (
    Any,
    Generic,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
EventT = TypeVar("EventT")


async def cancel_task(
    task: asyncio.Task | None, timeout: float = 1.0
) -> bool:
    """Safely cancel an asyncio.Task, suppressing CancelledError.

    Args:
        task: Target asyncio task. If None or already done, returns True immediately.
        timeout: Seconds to wait for task to complete its cancellation cleanup.

    Returns:
        True if the task is finished/cancelled, False if cancellation timed out.
    """
    if task is None or task.done():
        return True

    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return True
    except asyncio.CancelledError:
        return True
    except asyncio.TimeoutError:
        logger.warning("Task cancellation timed out after %s seconds: %s", timeout, task)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("Task %s raised error during cancellation: %s", task, exc)
        return True


async def cancel_tasks(
    tasks: Iterable[asyncio.Task | None], timeout: float = 1.0
) -> None:
    """Cancel multiple asyncio tasks concurrently with exception shielding.

    Args:
        tasks: Collection of tasks to terminate.
        timeout: Aggregate timeout for all tasks to finalize.
    """
    active_tasks = [t for t in tasks if t is not None and not t.done()]
    if not active_tasks:
        return

    for t in active_tasks:
        t.cancel()

    shielded = [asyncio.shield(t) for t in active_tasks]
    try:
        await asyncio.wait_for(
            asyncio.gather(*shielded, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Batch cancellation of %d tasks timed out.", len(active_tasks))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Error during batch task cancellation: %s", exc)


class InterruptibleQueue(asyncio.Queue, Generic[T]):
    """An asyncio Queue supporting atomic flushing and cancellation safety.

    Essential for speech and audio pipelines where an interruption must
    immediately discard all queued audio or tokens without deadlocking consumers.
    """

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize=maxsize)
        self._is_flushed: bool = False

    def flush(self) -> list[T]:
        """Atomically extract and discard all items currently in the queue.

        Returns:
            List of all discarded items.
        """
        items: list[T] = []
        while not self.empty():
            try:
                items.append(self.get_nowait())
                self.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
        return items

    async def get_cancellable(self) -> T:
        """Fetch an item from the queue with cancellation safety."""
        return await self.get()

    async def drain(self) -> AsyncIterator[T]:
        """Asynchronous generator yielding all remaining items until queue is empty."""
        while not self.empty():
            try:
                item = self.get_nowait()
                yield item
                self.task_done()
            except asyncio.QueueEmpty:
                break


class TaskTracker:
    """Tracks active background tasks with support for group cancellation.

    Automatically removes finished tasks using callbacks to prevent memory leaks.
    """

    def __init__(self, name: str = "default"):
        self.name: str = name
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def count(self) -> int:
        """Current count of active tracked tasks."""
        return len(self._tasks)

    @property
    def active_names(self) -> list[str]:
        """List of names of currently active tasks."""
        return list(self._tasks.keys())

    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        name: str | None = None,
    ) -> asyncio.Task:
        """Create, track, and return an asyncio Task from a coroutine.

        Args:
            coro: The coroutine to schedule.
            name: Human-readable identifier for tracking and selective cancellation.

        Returns:
            The created asyncio.Task.
        """
        task_id = name or f"task_{len(self._tasks)}_{id(coro)}"
        task = asyncio.create_task(coro, name=task_id)
        self._tasks[task_id] = task

        def _cleanup(t: asyncio.Task) -> None:
            self._tasks.pop(task_id, None)

        task.add_done_callback(_cleanup)
        return task

    def track(self, task: asyncio.Task, name: str | None = None) -> asyncio.Task:
        """Register an existing task into this tracker."""
        task_id = name or task.get_name() or f"task_{len(self._tasks)}_{id(task)}"
        self._tasks[task_id] = task

        def _cleanup(t: asyncio.Task) -> None:
            self._tasks.pop(task_id, None)

        task.add_done_callback(_cleanup)
        return task

    async def cancel_all(self, timeout: float = 1.0) -> None:
        """Cancel and await all currently registered tasks."""
        tasks_to_cancel = list(self._tasks.values())
        await cancel_tasks(tasks_to_cancel, timeout=timeout)
        self._tasks.clear()

    async def cancel_matching(self, prefix: str, timeout: float = 1.0) -> None:
        """Cancel and await all tasks whose name starts with the given prefix."""
        matching = [
            task for name, task in self._tasks.items() if name.startswith(prefix)
        ]
        await cancel_tasks(matching, timeout=timeout)


HandlerType = Callable[[Any], Coroutine[Any, Any, None]]


class AsyncEventBus:
    """Lightweight, typed asynchronous publish-subscribe event dispatcher.

    Allows decoupled communication between pipeline modules:
    VAD -> STT -> LLM -> ToolController -> Tools -> TTS.
    """

    def __init__(self):
        self._subscribers: dict[type[Any], set[HandlerType]] = defaultdict(set)
        self._lock: asyncio.Lock = asyncio.Lock()

    def subscribe(self, event_type: type[EventT], handler: HandlerType) -> Callable[[], None]:
        """Subscribe an async handler to a specific event type.

        Returns:
            An unsubscribe callable.
        """
        self._subscribers[event_type].add(handler)

        def _unsubscribe():
            self.unsubscribe(event_type, handler)

        return _unsubscribe

    def unsubscribe(self, event_type: type[EventT], handler: HandlerType) -> None:
        """Unsubscribe a previously registered handler."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribed handlers concurrently.

        Exceptions in handlers are caught and logged, preventing cascade failures.
        """
        event_type = type(event)
        handlers = list(self._subscribers.get(event_type, set()))

        # Also notify handlers registered for parent classes (e.g. BaseEvent)
        for registered_type, type_handlers in self._subscribers.items():
            if registered_type is not event_type and isinstance(event, registered_type):
                handlers.extend(type_handlers)

        if not handlers:
            return

        async def _safe_execute(handler: HandlerType) -> None:
            try:
                await handler(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Error executing handler %s for event %s",
                    handler.__name__,
                    event_type.__name__,
                )

        results = await asyncio.gather(
            *[_safe_execute(h) for h in handlers],
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res

    async def wait_for(
        self,
        event_type: type[EventT],
        predicate: Callable[[EventT], bool] | None = None,
        timeout: float | None = None,
    ) -> EventT:
        """Asynchronously wait for the next occurrence of an event matching predicate.

        Args:
            event_type: The event class to wait for.
            predicate: Optional filter function returning True for target event.
            timeout: Maximum seconds to wait.

        Returns:
            The matched event.

        Raises:
            asyncio.TimeoutError: If timeout expires before matching event arrives.
        """
        future: asyncio.Future[EventT] = asyncio.get_running_loop().create_future()

        async def _listener(event: EventT) -> None:
            if not future.done() and (predicate is None or predicate(event)):
                future.set_result(event)

        unsubscribe = self.subscribe(event_type, _listener)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            unsubscribe()

    def clear(self) -> None:
        """Remove all registered subscribers."""
        self._subscribers.clear()
