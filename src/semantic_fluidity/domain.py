"""
Domain inference and namespacing.

Every ingested document is tagged with a coarse *domain* before extraction, so
that invariants from unrelated fields never collide: a state variable named
``rate`` extracted from a genomics paper and one extracted from a game-engine
source file end up as ``genomics::rate`` and ``code:cpp::rate`` respectively
(see :mod:`src.semantic_fluidity.schema`).  Domains are inferred heuristically
from keyword frequency in the text plus the file's path; code files are always
keyed by their own ``code:<language>`` domain, independent of any prose
domains, since mixing source identifiers with natural-language vocabulary would
be meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from src.semantic_fluidity.documents import IngestedDocument

# Small built-in lexicon per prose domain.  Heuristic, not exhaustive -- the
# goal is "good enough to keep unrelated fields from colliding", not a full
# topic classifier.
DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "genomics": {
        "dna", "rna", "genome", "gene", "allele", "codon", "nucleotide",
        "chromosome", "mutation", "transcription", "genotype", "phenotype",
        "exon", "intron", "genomic",
    },
    "medicine": {
        "patient", "diagnosis", "dosage", "dose", "clinical", "symptom",
        "treatment", "therapy", "mg/kg", "contraindicated", "prescribed",
        "physician", "comorbidity",
    },
    "economics": {
        "gdp", "inflation", "equilibrium", "elasticity", "market", "supply",
        "demand", "interest rate", "utility", "macroeconomic", "fiscal",
        "monetary", "tariff", "recession",
    },
    "physics": {
        "velocity", "acceleration", "force", "energy", "momentum", "mass",
        "newton", "joule", "thermodynamic", "quantum", "relativity", "torque",
    },
    "game_engine": {
        "frame", "render", "shader", "entity", "sprite", "collider",
        "viewport", "raycast", "physics tick", "game loop", "actor",
        "skeletal", "tilemap",
    },
    "mathematics": {
        "theorem", "integral", "derivative", "matrix", "vector", "polynomial",
        "proof", "lemma", "axiom", "eigenvalue", "differential",
    },
}

DEFAULT_DOMAIN = "general"


def infer_domain(document: "IngestedDocument") -> str:
    """Infer a namespacing domain for ``document``.

    Code documents are always namespaced by their own ``code:<language>``
    format string.  Prose/JSON documents are scored against
    :data:`DOMAIN_KEYWORDS`; the highest-scoring domain wins, falling back to
    :data:`DEFAULT_DOMAIN` when nothing scores above zero.
    """
    if document.format.startswith("code:"):
        return document.format

    haystack = f"{document.path.stem} {document.text}".lower()
    best_domain = DEFAULT_DOMAIN
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(haystack.count(keyword) for keyword in keywords)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain
