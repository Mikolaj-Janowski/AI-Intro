"""Hand-written containers used by the search algorithms.

Nothing here relies on ``heapq``/``collections.deque`` so that the behaviour of
the frontier of each search method is fully visible in this project.
"""
from __future__ import annotations

from typing import Any, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class Queue(Generic[T]):
    """FIFO queue (BFS frontier) backed by a list with a moving head index.

    Pop is amortised O(1): the consumed prefix is dropped only occasionally.
    """

    __slots__ = ("_items", "_head")

    def __init__(self) -> None:
        self._items: List[Optional[T]] = []
        self._head = 0

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if self._head >= len(self._items):
            raise IndexError("pop from an empty queue")
        item = self._items[self._head]
        self._items[self._head] = None  # release the reference early
        self._head += 1
        if self._head >= 4096 and self._head * 2 >= len(self._items):
            del self._items[: self._head]
            self._head = 0
        return item  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._items) - self._head

    def __bool__(self) -> bool:
        return len(self) > 0


class Stack(Generic[T]):
    """LIFO stack (DFS frontier)."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)


class MinHeap(Generic[T]):
    """Binary min-heap keyed by an arbitrary comparable key (A*/Dijkstra frontier).

    Entries with equal keys are returned in insertion order (FIFO tie-break).
    """

    __slots__ = ("_data", "_counter")

    def __init__(self) -> None:
        self._data: List[Tuple[Any, int, T]] = []
        self._counter = 0

    def push(self, key: Any, item: T) -> None:
        self._data.append((key, self._counter, item))
        self._counter += 1
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Tuple[Any, T]:
        if not self._data:
            raise IndexError("pop from an empty heap")
        data = self._data
        last = data.pop()
        if not data:
            return last[0], last[2]
        top = data[0]
        data[0] = last
        self._sift_down(0)
        return top[0], top[2]

    def peek(self) -> Tuple[Any, T]:
        if not self._data:
            raise IndexError("peek at an empty heap")
        key, _, item = self._data[0]
        return key, item

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    # ------------------------------------------------------------ internals
    def _sift_up(self, i: int) -> None:
        data = self._data
        entry = data[i]
        while i > 0:
            parent = (i - 1) >> 1
            if data[parent][:2] <= entry[:2]:
                break
            data[i] = data[parent]
            i = parent
        data[i] = entry

    def _sift_down(self, i: int) -> None:
        data = self._data
        size = len(data)
        entry = data[i]
        while True:
            left = 2 * i + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if right < size and data[right][:2] < data[left][:2]:
                child = right
            if entry[:2] <= data[child][:2]:
                break
            data[i] = data[child]
            i = child
        data[i] = entry
