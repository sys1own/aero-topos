"""Genetic operators for the self-evolution engine: mutation and crossover."""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("evolution.genetic_operators")


class MutationEngine:
    """Applies random mutations to an individual's genome."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        evo = config.get("project", {}).get("evolutionary_bootstrap", {})
        # Copy so library genes can be appended without mutating shared config.
        self.mutation_vectors = dict(evo.get("mutation_vectors", {}))
        self.mutation_rate = 0.3
        self.library_genes: List[str] = []
        self.framework_genes: List[str] = []
        self._add_library_genes(config)
        self._add_framework_genes(config)

    def _add_library_genes(self, config: Dict[str, Any]) -> None:
        """Fold numerical-library choices into the genome (feature #2).

        When a ``[libraries]`` section is present, each tunable library becomes a
        categorical gene so the evolutionary engine can search combinations of
        BLAS/LAPACK/MPI/CUDA backends alongside the compiler knobs.
        """
        lib = config.get("libraries")
        if not isinstance(lib, dict):
            return

        genes: Dict[str, List[str]] = {}
        for key in ("blas", "lapack"):
            choice = str(lib.get(key, "none")).lower()
            if choice == "none":
                continue
            genes[key] = ["mkl", "openblas", "none"] if choice == "auto" else [choice, "none"]
        if lib.get("mpi"):
            flavor = lib.get("mpi_flavor")
            genes["mpi_flavor"] = [flavor, "none"] if flavor else ["openmpi", "mpich", "none"]
        if str(lib.get("cuda", "none")).lower() != "none":
            genes["cuda"] = ["auto", "none"]

        for key, values in genes.items():
            deduped: List[str] = []
            for v in values:
                if v not in deduped:
                    deduped.append(v)
            self.mutation_vectors[key] = deduped
            self.library_genes.append(key)

    def _add_framework_genes(self, config: Dict[str, Any]) -> None:
        """Fold physics-framework versions into the genome (feature #4).

        When a framework lists several candidate ``versions`` the engine may
        search them for the best performance.
        """
        if not config.get("frameworks"):
            return
        try:
            from src.build.framework_integration import FrameworkIntegration

            space = FrameworkIntegration(config).genome_space()
        except ImportError:
            return
        except Exception as exc:
            logger.warning("Failed to load framework genome space: %s", exc)
            return
        for key, values in space.items():
            self.mutation_vectors[key] = list(values)
            self.framework_genes.append(key)

    def apply_mutations(self, workspace: Any, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Return the workspace path unchanged (mutations are config-level, not source-level)."""
        return workspace

    @staticmethod
    def _sample(spec: Any) -> Any:
        """Draw one value from a gene *spec*, supporting both schema dialects.

        Accepts either the legacy ``{"range": [lo, hi], "step": s}`` form or the
        unified ``{"min": lo, "max": hi, "step": s, "type": "int"|"float",
        "choices": [...]}`` schema shared with ``evolve.py``. A bare list is a
        categorical gene; any other scalar is returned verbatim. Returns ``None``
        when the spec carries no sampleable range so the caller can skip it.
        """
        if isinstance(spec, list):
            return random.choice(spec) if spec else None
        if not isinstance(spec, dict):
            return spec

        if spec.get("choices"):
            return random.choice(list(spec["choices"]))

        # Resolve bounds from either dialect.
        if "range" in spec:
            low, high = spec["range"]
        elif "min" in spec and "max" in spec:
            low, high = spec["min"], spec["max"]
        else:
            return None

        gene_type = spec.get("type") or ("int" if isinstance(low, int) and isinstance(high, int) else "float")
        step = spec.get("step")
        if gene_type == "float":
            if step:
                ticks = max(int(round((high - low) / step)), 0)
                return low + random.randint(0, ticks) * step
            return random.uniform(low, high)
        # Integer / discrete-stepped gene.
        istep = max(int(step or 1), 1)
        values = list(range(int(low), int(high) + 1, istep))
        return random.choice(values) if values else spec.get("default", low)

    def mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        mutated = copy.deepcopy(genome)
        for key, spec in self.mutation_vectors.items():
            if random.random() > self.mutation_rate:
                continue
            sampled = self._sample(spec)
            if sampled is not None:
                mutated[key] = sampled
        return mutated

    def generate_random(self) -> Dict[str, Any]:
        genome: Dict[str, Any] = {}
        for key, spec in self.mutation_vectors.items():
            sampled = self._sample(spec)
            if sampled is not None:
                genome[key] = sampled
            else:
                # Opaque scalar gene: carry its default (dict) or value (scalar).
                genome[key] = spec.get("default") if isinstance(spec, dict) else spec
        return genome


class CrossoverEngine:
    """Performs crossover between two parent genomes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        child: Dict[str, Any] = {}
        all_keys = set(parent_a.keys()) | set(parent_b.keys())
        for key in all_keys:
            if key in parent_a and key in parent_b:
                child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]
            elif key in parent_a:
                child[key] = parent_a[key]
            else:
                child[key] = parent_b[key]
        return child
