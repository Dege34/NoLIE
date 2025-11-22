# Contributing to Deepfake Forensics

Thank you for your interest in contributing to Deepfake Forensics! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/deepfake-forensics.git
   cd deepfake-forensics
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

5. Run tests to ensure everything works:
   ```bash
   python -m deepfake_forensics.cli test
   ```

## Development Workflow

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical bug fixes

### Making Changes

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards
3. Write tests for new functionality
4. Run the test suite:
   ```bash
   python -m deepfake_forensics.cli test
   ```

5. Run linting and formatting:
   ```bash
   python -m deepfake_forensics.cli lint
   python -m deepfake_forensics.cli format
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a pull request to `develop`

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Keep functions small and focused
- Use meaningful variable names

### Code Formatting

We use `black` for code formatting and `ruff` for linting:

```bash
# Format code
python -m deepfake_forensics.cli format

# Run linting
python -m deepfake_forensics.cli lint
```

### Type Hints

Use type hints for all function parameters and return values:

```python
def process_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Process an image with the given size."""
    # Implementation
    return processed_image
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int = 100
) -> Dict[str, float]:
    """Train a model on the given dataloader.
    
    Args:
        model: The model to train
        dataloader: The training data
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is not positive
    """
    # Implementation
```

## Testing

### Writing Tests

- Write unit tests for all new functionality
- Aim for 80%+ code coverage
- Use descriptive test names
- Test edge cases and error conditions

### Running Tests

```bash
# Run all tests
python -m deepfake_forensics.cli test

# Run with coverage
python -m deepfake_forensics.cli test --coverage

# Run specific test file
python -m deepfake_forensics.cli test --test-dir tests/test_models.py
```

### Test Structure

```python
def test_model_forward():
    """Test model forward pass."""
    # Arrange
    model = XceptionNet(num_classes=2)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Act
    output = model(input_tensor)
    
    # Assert
    assert output.shape == (1, 2)
    assert torch.allclose(output.sum(dim=1), torch.ones(1))
```

## Documentation

### Code Documentation

- Document all public APIs
- Include examples in docstrings
- Keep documentation up to date

### README Updates

- Update README.md for new features
- Include usage examples
- Update installation instructions if needed

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run linting and formatting
3. Update documentation if needed
4. Add tests for new functionality
5. Update CHANGELOG.md if applicable

### PR Description

Include:
- Description of changes
- Motivation for changes
- Testing performed
- Screenshots (if applicable)
- Breaking changes (if any)

### Review Process

- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and small
- Rebase on latest develop before merging

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Steps

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR to `main`
4. Tag the release
5. Publish to PyPI

## Getting Help

- Check existing issues and discussions
- Join our Discord server
- Email: team@deepfake-forensics.com

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Deepfake Forensics!
