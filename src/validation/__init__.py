"""Automated physical validation.

Runs a benchmark/validation suite against the built artefacts, compares results
to known-good references within a tolerance, and reports pass/fail.  When used
as a gatekeeper, only candidates that pass validation are admitted to the Pareto
front.
"""

from src.validation.validator import ValidationCaseResult, ValidationReport, Validator

__all__ = ["Validator", "ValidationReport", "ValidationCaseResult"]
