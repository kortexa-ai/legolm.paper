"""
Lightweight per-step CSV logger for paper experiments.
"""

from __future__ import annotations

import csv
import os


class StepLogger:
    """Append-only CSV logger. Creates parent dirs automatically."""

    def __init__(self, path: str):
        self.path = path
        self._writer = None
        self._file = None
        self._fields = None

    def log(self, **kwargs) -> None:
        if self._writer is None:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._file = open(self.path, "w", newline="")
            self._fields = list(kwargs.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
