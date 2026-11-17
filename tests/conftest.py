"""Shared pytest configuration for the mira test suite."""

import matplotlib

# Tests must run without a display (CLAUDE.md, testing conventions).
matplotlib.use("Agg")
