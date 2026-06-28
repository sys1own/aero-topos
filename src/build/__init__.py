"""Build-system extensions for large-scale physics simulation projects.

This package houses the capabilities that let the Aero Multi-Tool scale from a
single-machine source optimiser to a coordinator for multi-million-line,
multi-language numerical codebases:

* :mod:`src.build.library_tuner`  -- BLAS/LAPACK/MPI/CUDA detection + flags.
* :mod:`src.build.gpu_pipeline`   -- GPU kernel compilation (nvcc/hipcc).
* :mod:`src.build.distributed`    -- distributed compilation coordination.

Every module degrades gracefully: if a library, compiler or worker backend is
absent, the feature reports "unavailable" instead of raising, so the tool stays
usable on a laptop with none of the heavy infrastructure installed.
"""

from src.build.library_tuner import DetectedLibrary, LibraryTuner

__all__ = ["DetectedLibrary", "LibraryTuner"]
