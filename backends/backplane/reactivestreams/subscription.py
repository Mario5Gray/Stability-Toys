from abc import ABC, abstractmethod


class Subscription(ABC):
    @abstractmethod
    def request(self, n: int) -> None:
        """Signal demand for up to n more items (n may be a large sentinel for unbounded)."""

    @abstractmethod
    def cancel(self) -> None:
        """Stop the stream; no further on_next/on_complete/on_error after this returns."""
