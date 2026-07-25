from abc import ABC, abstractmethod

from .subscription import Subscription


class Subscriber(ABC):
    @abstractmethod
    def on_subscribe(self, subscription: Subscription) -> None: ...

    @abstractmethod
    def on_next(self, value) -> None: ...

    @abstractmethod
    def on_error(self, error: Exception) -> None: ...

    @abstractmethod
    def on_complete(self) -> None: ...
