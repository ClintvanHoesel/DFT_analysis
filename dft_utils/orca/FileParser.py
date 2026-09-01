from abc import ABC, abstractmethod
from pathlib import Path


class FileParser(ABC):
    """Abstract base for all file-type parsers."""

    def __init__(self, path: str):
        """Initialize the object."""
        self.path = Path(path)

    @abstractmethod
    def parse(self):
        """Implement parsing logic."""
        pass
