# Contributing to Credit Card Fraud Detection MLOps

Thank you for your interest in contributing! We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](../../issues)
2. If not, create a new issue with:
   - Clear descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Enhancements

1. Check existing [Issues](../../issues) and [Pull Requests](../../pulls)
2. Create a new issue describing:
   - The enhancement/feature
   - Why it would be useful
   - Possible implementation approach

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-mlops.git
   cd fraud-detection-mlops
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check code style
   black src/ api/ app.py
   flake8 src/ api/ app.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure all checks pass

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for functions and classes
- Keep functions focused and modular
- Use meaningful variable names

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update configuration examples if needed

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

Feel free to open an issue or reach out to maintainers.

Thank you for contributing! 🎉
