"""
Unit tests for async utilities in src/core/async_utils.py.
"""

import asyncio

import pytest

from src.core.async_utils import (
    AsyncEventBus,
    InterruptibleQueue,
    TaskTracker,
    cancel_task,
    cancel_tasks,
)
from src.core.events import (
    AudioInputEvent,
    BaseEvent,
    InterruptionEvent,
    TranscriptDeltaEvent,
)


@pytest.mark.asyncio
async def test_cancel_task_active():
    """Verify clean cancellation of a running asyncio Task."""
    async def _long_worker():
        await asyncio.sleep(10.0)

    task = asyncio.create_task(_long_worker())
    success = await cancel_task(task, timeout=0.5)

    assert success is True
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_task_none_and_done():
    """cancel_task handles None and already completed tasks safely."""
    assert await cancel_task(None) is True

    async def _instant():
        return 42

    task = asyncio.create_task(_instant())
    await task
    assert await cancel_task(task) is True


@pytest.mark.asyncio
async def test_cancel_tasks_batch():
    """Verify batch cancellation of multiple running tasks."""
    async def _worker():
        await asyncio.sleep(10.0)

    tasks = [asyncio.create_task(_worker()) for _ in range(5)]
    await cancel_tasks(tasks, timeout=0.5)

    for t in tasks:
        assert t.cancelled() or t.done()


@pytest.mark.asyncio
async def test_interruptible_queue_put_get():
    """Basic put and get functionality."""
    q: InterruptibleQueue[str] = InterruptibleQueue()
    await q.put("item1")
    await q.put("item2")

    res = await q.get_cancellable()
    assert res == "item1"


@pytest.mark.asyncio
async def test_interruptible_queue_flush():
    """Atomically flush queued items upon interruption."""
    q: InterruptibleQueue[int] = InterruptibleQueue()
    for i in range(10):
        await q.put(i)

    assert q.qsize() == 10
    discarded = q.flush()

    assert discarded == list(range(10))
    assert q.empty()
    assert q.qsize() == 0


@pytest.mark.asyncio
async def test_interruptible_queue_drain():
    """Drain remaining items via async iterator."""
    q: InterruptibleQueue[str] = InterruptibleQueue()
    await q.put("a")
    await q.put("b")

    items = []
    async for item in q.drain():
        items.append(item)

    assert items == ["a", "b"]
    assert q.empty()


@pytest.mark.asyncio
async def test_task_tracker_spawn_and_autocleanup():
    """TaskTracker should track active tasks and automatically discard finished ones."""
    tracker = TaskTracker(name="test_tracker")

    async def _quick_worker():
        await asyncio.sleep(0.01)

    task = tracker.spawn(_quick_worker(), name="quick_1")
    assert tracker.count == 1
    assert "quick_1" in tracker.active_names

    await task
    # Allow callback to run
    await asyncio.sleep(0.01)
    assert tracker.count == 0


@pytest.mark.asyncio
async def test_task_tracker_cancel_all():
    """TaskTracker cancel_all cancels all in-flight tasks."""
    tracker = TaskTracker(name="test_tracker")

    async def _long_worker():
        await asyncio.sleep(10.0)

    tracker.spawn(_long_worker(), name="long_1")
    tracker.spawn(_long_worker(), name="long_2")
    assert tracker.count == 2

    await tracker.cancel_all(timeout=0.5)
    assert tracker.count == 0


@pytest.mark.asyncio
async def test_task_tracker_cancel_matching():
    """TaskTracker cancel_matching selectively cancels by prefix."""
    tracker = TaskTracker(name="test_tracker")

    async def _long_worker():
        await asyncio.sleep(10.0)

    tracker.spawn(_long_worker(), name="tts_play_1")
    tracker.spawn(_long_worker(), name="tts_play_2")
    tracker.spawn(_long_worker(), name="asr_stream_1")
    assert tracker.count == 3

    await tracker.cancel_matching("tts_", timeout=0.5)
    assert tracker.count == 1
    assert tracker.active_names == ["asr_stream_1"]

    await tracker.cancel_all(timeout=0.5)


@pytest.mark.asyncio
async def test_async_event_bus_publish_subscribe():
    """EventBus delivers published events to matching subscribers."""
    bus = AsyncEventBus()
    received = []

    async def _on_transcript(event: TranscriptDeltaEvent):
        received.append(event.text)

    unsubscribe = bus.subscribe(TranscriptDeltaEvent, _on_transcript)

    await bus.publish(TranscriptDeltaEvent(text="first"))
    await bus.publish(TranscriptDeltaEvent(text="second"))

    assert received == ["first", "second"]

    # Test unsubscribe
    unsubscribe()
    await bus.publish(TranscriptDeltaEvent(text="third"))
    assert received == ["first", "second"]


@pytest.mark.asyncio
async def test_async_event_bus_polymorphic_dispatch():
    """Subscribing to BaseEvent receives all derived events."""
    bus = AsyncEventBus()
    received_types = []

    async def _on_any_event(event: BaseEvent):
        received_types.append(type(event))

    bus.subscribe(BaseEvent, _on_any_event)

    await bus.publish(InterruptionEvent(reason="test"))
    await bus.publish(AudioInputEvent(pcm_data=b""))

    assert received_types == [InterruptionEvent, AudioInputEvent]


@pytest.mark.asyncio
async def test_async_event_bus_exception_isolation():
    """An exception in one event handler must not prevent others from running."""
    bus = AsyncEventBus()
    order = []

    async def _failing_handler(event: InterruptionEvent):
        order.append("fail")
        raise RuntimeError("Handler explosion")

    async def _healthy_handler(event: InterruptionEvent):
        order.append("healthy")

    bus.subscribe(InterruptionEvent, _failing_handler)
    bus.subscribe(InterruptionEvent, _healthy_handler)

    await bus.publish(InterruptionEvent(reason="isolated"))
    assert "fail" in order
    assert "healthy" in order


@pytest.mark.asyncio
async def test_async_event_bus_wait_for():
    """wait_for resolves when matching event arrives."""
    bus = AsyncEventBus()

    async def _delayed_publish():
        await asyncio.sleep(0.02)
        await bus.publish(TranscriptDeltaEvent(text="hello", is_final=True))

    asyncio.create_task(_delayed_publish())

    event = await bus.wait_for(
        TranscriptDeltaEvent,
        predicate=lambda e: e.is_final is True,
        timeout=1.0,
    )
    assert event.text == "hello"
    assert event.is_final is True


@pytest.mark.asyncio
async def test_async_event_bus_wait_for_timeout():
    """wait_for raises TimeoutError when event does not arrive within deadline."""
    bus = AsyncEventBus()
    with pytest.raises(asyncio.TimeoutError):
        await bus.wait_for(InterruptionEvent, timeout=0.05)
