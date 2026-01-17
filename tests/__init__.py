"""
LCR Test Suite

This package contains unit tests, integration tests, and regression tests
for the LCR (Latent Conflict Refinement) system.

Test Organization:
    tests/
    ├── unit/           - Fast, isolated unit tests
    ├── integration/    - End-to-end integration tests
    ├── fixtures/       - Shared test data and fixtures
    └── conftest.py     - Pytest configuration and fixtures

Running Tests:
    pytest tests/                    # All tests
    pytest tests/unit/               # Unit tests only
    pytest tests/ -v --cov=src       # With coverage
    pytest tests/ -k "detector"      # Filter by name
"""
