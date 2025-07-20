# Contributing to Uncertainty-Aware Intrusion Detection System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/uncertainty-ids.git
   cd uncertainty-ids
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify installation**
   ```bash
   pytest tests/
   ```

## Code Style

We use several tools to maintain code quality:

### Formatting
- **Black**: Code formatter
  ```bash
  black uncertainty_ids/ tests/
  ```

- **isort**: Import sorter
  ```bash
  isort uncertainty_ids/ tests/
  ```

### Linting
- **flake8**: Style guide enforcement
  ```bash
  flake8 uncertainty_ids/ tests/
  ```

- **mypy**: Type checking
  ```bash
  mypy uncertainty_ids/
  ```

### Pre-commit Hooks

We use pre-commit hooks to automatically run these tools:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uncertainty_ids

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run performance benchmarks
pytest -m benchmark
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_model_prediction_with_uncertainty`
- Include both unit tests and integration tests
- Test edge cases and error conditions
- Use fixtures for common test data

Example test structure:
```python
import pytest
from uncertainty_ids.models import BayesianEnsembleIDS

class TestBayesianEnsembleIDS:
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = BayesianEnsembleIDS(n_ensemble=5)
        assert model.n_ensemble == 5
    
    def test_prediction_output_format(self):
        """Test prediction output has correct format."""
        # Test implementation
        pass
```

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Document all public APIs
- Update docstrings when changing function signatures
- Add type hints to all functions

Example docstring format:
```python
def predict_with_uncertainty(self, network_prompt: torch.Tensor, 
                           query_flow: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Make predictions with comprehensive uncertainty quantification.
    
    Args:
        network_prompt: Historical network flows (batch_size, seq_len, n_features)
        query_flow: Current flow to classify (batch_size, n_features)
        
    Returns:
        Dictionary containing predictions, probabilities, and uncertainty estimates
        
    Raises:
        ValueError: If input tensors have incompatible shapes
        
    Example:
        >>> model = BayesianEnsembleIDS()
        >>> results = model.predict_with_uncertainty(prompt, query)
        >>> print(f"Prediction: {results['predictions']}")
    """
```

## Issue Reporting

### Bug Reports

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

Use our bug report template when creating issues.

### Feature Requests

We welcome feature requests! Please:

- Explain the motivation for the feature
- Describe the proposed solution
- Consider alternative solutions
- Provide examples of how it would be used

## Code Review Process

### For Contributors

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Update documentation
5. Run the full test suite
6. Submit a pull request

### For Reviewers

- Check that tests pass
- Verify code follows style guidelines
- Ensure documentation is updated
- Test the changes locally
- Provide constructive feedback

## Release Process

1. Update version numbers in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create a release branch
4. Run full test suite
5. Create a pull request to `main`
6. After merge, create a GitHub release
7. Publish to PyPI

## Community Guidelines

### Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Communication

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Ask questions when unsure

## Getting Help

- **Documentation**: https://uncertainty-ids.readthedocs.io/
- **Issues**: https://github.com/research-team/uncertainty-ids/issues
- **Discussions**: https://github.com/research-team/uncertainty-ids/discussions
- **Email**: research@example.com

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask! Create an issue or start a discussion if you have any questions about contributing.

---

Thank you for contributing to the Uncertainty-Aware Intrusion Detection System! 🎉
