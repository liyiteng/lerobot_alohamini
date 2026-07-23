"""Name-to-class registry for data-engine components."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, type[Any]]] = defaultdict(dict)

    def register(self, group: str, name: str, cls: type[T]) -> type[T]:
        key = _normalize(name)
        if key in self._items[group] and self._items[group][key] is not cls:
            raise KeyError(f"{group!r} component {name!r} is already registered.")
        self._items[group][key] = cls
        return cls

    def decorator(self, group: str, name: str) -> Callable[[type[T]], type[T]]:
        def _wrap(cls: type[T]) -> type[T]:
            return self.register(group, name, cls)

        return _wrap

    def get(self, group: str, name: str) -> type[Any]:
        key = _normalize(name)
        try:
            return self._items[group][key]
        except KeyError as exc:
            known = ", ".join(sorted(self._items[group])) or "<none>"
            raise KeyError(
                f"Unknown {group} component {name!r}. Registered: {known}"
            ) from exc

    def build(self, group: str, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(group, name)(*args, **kwargs)

    def names(self, group: str) -> list[str]:
        return sorted(self._items[group])


def _normalize(name: str) -> str:
    return name.strip().lower()


REGISTRY = ComponentRegistry()

register_loader = lambda name: REGISTRY.decorator("loader", name)
register_randomizer = lambda name: REGISTRY.decorator("randomizer", name)
register_planner = lambda name: REGISTRY.decorator("planner", name)
register_renderer = lambda name: REGISTRY.decorator("renderer", name)
register_writer = lambda name: REGISTRY.decorator("writer", name)


def build_loader(name: str, *args: Any, **kwargs: Any) -> Any:
    return REGISTRY.build("loader", name, *args, **kwargs)


def build_randomizer(name: str, *args: Any, **kwargs: Any) -> Any:
    return REGISTRY.build("randomizer", name, *args, **kwargs)


def build_planner(name: str, *args: Any, **kwargs: Any) -> Any:
    return REGISTRY.build("planner", name, *args, **kwargs)


def build_renderer(name: str, *args: Any, **kwargs: Any) -> Any:
    return REGISTRY.build("renderer", name, *args, **kwargs)


def build_writer(name: str, *args: Any, **kwargs: Any) -> Any:
    return REGISTRY.build("writer", name, *args, **kwargs)
