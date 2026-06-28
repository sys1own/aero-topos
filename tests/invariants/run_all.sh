#!/usr/bin/env bash
# Execute all invariants tests in the topological engine
echo "Running Aero Topos Invariant Suite..."
python test_topos.py
exit $?
