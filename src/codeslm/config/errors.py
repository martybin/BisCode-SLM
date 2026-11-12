"""Configuration errors."""

from pathlib import Path


class ConfigError(Exception):
    """One or more configuration checks failed.

    ``errors`` holds the individual messages. The exception string joins them
    so a traceback is readable without inspecting the attribute.
    """

    def __init__(self, errors: list[str], path: Path | None = None) -> None:
        self.path = path
        self.errors = errors
        prefix = f"{path}: " if path is not None else ""
        super().__init__(prefix + "; ".join(errors))
