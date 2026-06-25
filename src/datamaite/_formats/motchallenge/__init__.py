"""MOTChallenge format package."""

from datamaite._formats.motchallenge.loader import MotChallengeLoader, load_motchallenge
from datamaite._formats.motchallenge.writer import MotChallengeWriter

__all__ = ["MotChallengeLoader", "MotChallengeWriter", "load_motchallenge"]
