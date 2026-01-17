# Contributing to LCR

**Status**: Active Development

Thank you for your interest in contributing to the Latent Conflict Resolution project!

---

## Contribution Philosophy

This repository serves dual purposes:
1. **Paper Reproduction**: Primary goal - maintaining exact reproducibility
2. **Research Extension**: Secondary goal - enabling future research

## Development Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Installation

```bash
# Clone repository
git clone [ANONYMOUS_REPOSITORY_URL]
cd lcr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detector.py -v
```

## Code Style

We use the following tools for code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src/ scripts/
isort src/ scripts/

# Check linting
flake8 src/ scripts/

# Type checking
mypy src/
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature`
3. **Make changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Format code**: `black . && isort .`
6. **Commit** with clear message
7. **Push** and create PR

### Commit Messages

Follow conventional commits:
```
feat: add new baseline implementation
fix: correct L4 accuracy calculation
docs: update README with examples
refactor: simplify refinement loop
test: add detector unit tests
```

## Project Structure

```
lcr/
├── src/              # Core implementation
├── scripts/          # Training/evaluation scripts
├── configs/          # Configuration files
├── data/             # Datasets
├── checkpoints/      # Model weights
├── tests/            # Unit tests
└── docs/             # Documentation
```

## Adding New Components

### New Baseline
1. Add implementation to `scripts/baselines/`
2. Add config to `configs/baselines/`
3. Add to baseline comparison script
4. Document in `scripts/README.md`

### New Model Variant
1. Add to `src/models/`
2. Create config in `configs/`
3. Add training script if needed
4. Update `src/models/__init__.py`

## Questions?

- Open an issue for bugs or feature requests
- See the project's GitHub Issues page for discussions
