"""Explicit bounded cache for classical planning artifacts, never backend objects."""

from collections import OrderedDict
from dataclasses import dataclass
from sys import getsizeof
from threading import Lock


def object_bytes(root):
    """Account reachable Python objects once per entry; excludes allocator overhead."""
    seen, stack, size = set(), [root], 0
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        size += getsizeof(value)
        if isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            stack.extend(value)
        elif hasattr(value, "__dict__"):
            stack.append(vars(value))
    return size


@dataclass(frozen=True)
class CacheInfo:
    hits: int
    misses: int
    evictions: int
    entries: int
    max_entries: int
    accounted_bytes: int = 0
    max_bytes: int | None = None
    oversized_rejections: int = 0


class PlanningCache:
    """LRU storage with optional accounted Python-object byte and entry limits.

    Store immutable request analysis only. Selection, model validation and memory
    checks run again on every call. Concurrent misses may repeat pure analysis.
    """

    def __init__(self, max_entries=128, *, max_bytes=None):
        """Create a bounded planning cache with optional serialized-size limits."""
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max_bytes must be a positive integer or None")
        self._max_bytes = max_bytes
        self._sizes = {}
        self._bytes = self._oversized = 0
        self._entries = OrderedDict()
        self._hits = self._misses = self._evictions = 0
        self._lock = Lock()

    def _get(self, key):
        """Return a cached value and update recency statistics, if present."""
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
                self._entries.move_to_end(key)
            return value

    def _put(self, key, value):
        """Store a value while evicting entries that exceed configured limits."""
        size = object_bytes((key, value)) if self._max_bytes is not None else 0
        with self._lock:
            if self._max_bytes is not None and size > self._max_bytes:
                self._oversized += 1
                return
            self._bytes -= self._sizes.get(key, 0)
            self._entries[key] = value
            self._sizes[key] = size
            self._bytes += size
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries or (
                self._max_bytes is not None and self._bytes > self._max_bytes
            ):
                expired, _ = self._entries.popitem(last=False)
                self._bytes -= self._sizes.pop(expired)
                self._evictions += 1

    def clear(self):
        """Release entries and reset statistics; existing Plans remain valid."""
        with self._lock:
            self._entries.clear()
            self._sizes.clear()
            self._bytes = self._oversized = 0
            self._hits = self._misses = self._evictions = 0

    def info(self):
        """Return cache size, byte usage, and hit/miss statistics."""
        with self._lock:
            return CacheInfo(
                self._hits,
                self._misses,
                self._evictions,
                len(self._entries),
                self._max_entries,
                self._bytes,
                self._max_bytes,
                self._oversized,
            )
