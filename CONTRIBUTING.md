# Contributing to Tianshou-vec

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the Tianshou-vec project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing opinions, experiences, and viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account
- Familiarity with reinforcement learning and PyTorch (helpful)

### Fork and Clone

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Tianshou-vec.git
   cd Tianshou-vec
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/minakhoshbazm/Tianshou-vec.git
   ```

## Development Setup

### Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy

# Create necessary directories
mkdir -p logs plots src/{core,agents,utils,data} tests
```

### Verify Installation

```bash
# Run a quick test
python -m pytest tests/ -v
```

## Making Changes

### Create a Feature Branch

```bash
# Update master branch
git checkout master
git pull upstream master

# Create feature branch with descriptive name
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
# or for documentation:
git checkout -b docs/documentation-topic
```

### Branch Naming Convention

- `feature/` - New features or enhancements
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements
- `perf/` - Performance improvements

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications. Use the following tools:

#### Black (Code Formatter)

```bash
# Format all Python files
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

**Black Configuration**: Default settings (line length: 88 characters)

#### Flake8 (Linter)

```bash
# Check for style issues
flake8 src/ tests/ --max-line-length=88
```

#### Type Hints

Use type hints for all functions:

```python
from typing import List, Dict, Tuple, Optional

def compute_reward(
    network: 'Network',
    action: int,
    tasks: List['Task']
) -> Tuple[float, int, float, float]:
    \"\"\"
    Compute reward for an action.
    
    Args:
        network: The network environment
        action: Action ID
        tasks: List of unprocessed tasks
    
    Returns:
        Tuple of (reward, penalty_count, cost, energy)
    \"\"\"
    pass
```

### Documentation Style

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    \"\"\"
    Brief one-line description.
    
    Extended description explaining the function in more detail.
    Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something is wrong with inputs
        RuntimeError: When computation fails
    
    Examples:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    \"\"\"
    pass
```

## Testing

### Writing Tests

Place tests in the `tests/` directory with names matching `test_*.py`:

```python
# tests/test_dqn_agent.py
import pytest
from src.agents.dqn_agent import DQNAgent

class TestDQNAgent:
    \"\"\"Test suite for DQN Agent.\"\"\"
    
    @pytest.fixture
    def agent(self):
        \"\"\"Create a DQN agent for testing.\"\"\"
        return DQNAgent()
    
    def test_agent_initialization(self, agent):
        \"\"\"Test that agent initializes correctly.\"\"\"
        assert agent is not None
        assert agent.policy is not None
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dqn_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow" -v
```

### Test Coverage

Aim for at least 80% code coverage:

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

## Submitting Changes

### Before You Submit

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run linting and formatting**:
   ```bash
   black src/ tests/
   flake8 src/ tests/ --max-line-length=88
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Check type hints** (optional but recommended):
   ```bash
   mypy src/ --ignore-missing-imports
   ```

## Commit Messages

Write clear, descriptive commit messages following this format:

```
<type>: <subject>

<body>

<footer>
```

### Type

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `ci`: Changes to CI configuration files and scripts

### Subject Line

- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period (.) at the end
- Limit to 50 characters

### Body

- Use imperative mood
- Include motivation for the change
- Contrast with previous behavior
- Limit each line to 72 characters

### Examples

```
feat: add prioritized experience replay buffer

Implement PER (Prioritized Experience Replay) for better
sample efficiency. Uses TD-error as priority metric.

References: #42
```

```
fix: correct reward computation for local offloading

Previously, local energy computation was using wrong
capability value. Now uses vehicle_capability correctly.

Fixes #38
```

## Pull Request Process

### Before Creating PR

1. Update master branch:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. Make sure all checks pass locally:
   ```bash
   black --check src/ tests/
   flake8 src/ tests/ --max-line-length=88
   pytest tests/ -v
   ```

### Create Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open PR on GitHub** with a clear title and description:
   ```
   # Title (use same format as commit messages)
   feat: implement Rainbow DQN with all 6 improvements
   
   # Description
   ## What does this PR do?
   Implements Rainbow DQN combining:
   - Double DQN
   - Prioritized Experience Replay
   - Dueling Networks
   - Multi-step Returns
   - Distributional RL (C51)
   - Noisy Networks
   
   ## How to test?
   Run: `python src/agents/rainbow_agent.py --epoch 10`
   
   ## Related Issues
   Fixes #45
   
   ## Screenshots/Results (if applicable)
   Training converges 2x faster than DQN baseline.
   ```

### PR Checklist

- [ ] I have read the CONTRIBUTING.md file
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] My code follows the style guidelines of this project

### PR Review Process

1. At least one maintainer will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Expected behavior
4. Actual behavior

## Environment
- Python version: 3.9
- PyTorch version: 1.10
- Tianshou version: 0.4.8
- OS: Ubuntu 20.04

## Logs/Error Messages
\`\`\`
error traceback here
\`\`\`

## Possible Solutions
Any ideas on how to fix this?
```

### Feature Requests

```markdown
## Is your feature request related to a problem?
Clear description of the problem

## Describe the Solution
What you want to happen

## Describe Alternatives
Other solutions or features you've considered

## Additional Context
Any other context or screenshots
```

## Questions?

If you have questions:
1. Check the README.md and existing documentation
2. Search existing GitHub issues
3. Open a new GitHub discussion (preferred for questions)
4. Contact maintainers if needed

## Additional Notes

### Performance Considerations

When adding new features:
- Vectorized operations are preferred (NumPy, PyTorch)
- Avoid Python loops in computationally intensive code
- Profile code with large-scale environments

### Algorithm Implementation

When implementing new RL algorithms:
- Follow Tianshou's policy interface
- Include clear mathematical formulations in docstrings
- Add reference papers or blogs
- Include test comparisons with reference implementations

### VANET/Network Simulation

When modifying network simulation:
- Ensure backward compatibility with existing traces
- Document any changes to network parameters
- Test with actual VANET datasets

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- GitHub contributor list
- Project release notes

Thank you for contributing to Tianshou-vec! 🚀
