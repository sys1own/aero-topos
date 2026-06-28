"""Mock physics-simulator orchestrator: compute pi to double precision.

This is the runnable entry point for the ``physics_simulator`` example. It
demonstrates an end-to-end Aero Future build target: a self-contained numerical
kernel whose result is validated against the IEEE-754 ``f64`` reference value.

Run it directly from the example directory::

    python -m src.python.orchestrator
"""

from __future__ import annotations

import math


def compute_pi() -> float:
    """Compute pi from the arctangent identity ``pi = 4 * arctan(1)``.

    Using the platform's correctly-rounded ``math.atan`` yields the nearest
    representable ``f64`` to pi, so the result is bit-for-bit identical to the
    IEEE-754 reference ``math.pi`` -- exactly what the precision shield expects
    of a protected fundamental constant.
    """
    return 4.0 * math.atan(1.0)


def main() -> int:
    computed = compute_pi()
    reference = math.pi
    matches = computed == reference

    print("Aero Future :: physics_simulator :: pi validation")
    print(f"  computed pi : {computed!r}")
    print(f"  reference   : {reference!r}")
    print(f"  ulp delta   : {abs(computed - reference)!r}")
    print(f"  matches f64 : {matches}")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
