"""Mixed / enhanced precision selection.

Chooses per-zone floating-point precision (double / quad / arbitrary), derives
the compiler flags and type mappings for each language, and -- when
``auto_detect_need`` is set -- heuristically flags UAST regions (iterative
solvers, transcendental functions, ill-conditioned linear algebra) that benefit
from higher precision.
"""

from src.precision.selector import PrecisionRecommendation, PrecisionSelector

__all__ = ["PrecisionRecommendation", "PrecisionSelector"]
