"""Shared types for databridge."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class DatasetFormat(enum.Enum):
    """Supported dataset formats."""

    HMIE = "hmie"


class Severity(enum.Enum):
    """Severity level for a validation finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """A single validation finding."""

    severity: Severity
    path: Path
    check: str
    message: str


@dataclass
class ValidationResult:
    """Outcome of validating one dataset."""

    dataset_path: Path
    dataset_format: DatasetFormat
    passed: bool = True
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.dataset_path}",
            *[f"  [{f.severity.value.upper()}] {f.path}: {f.message}" for f in self.findings],
        ]
        return "\n".join(lines)
