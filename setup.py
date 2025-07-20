"""
Setup script for Uncertainty-Aware Intrusion Detection System.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Core requirements
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.62.0",
    "pydantic>=1.8.0",
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "python-multipart>=0.0.5",
    "psutil>=5.8.0",
    "joblib>=1.1.0",
    "pyyaml>=6.0",
    "requests>=2.25.0",
]

# Development requirements
dev_requires = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "pytest-mock>=3.6.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "isort>=5.9.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "nbsphinx>=0.8.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "ipykernel>=6.0.0",
]

# Documentation requirements
docs_requires = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "nbsphinx>=0.8.0",
    "myst-parser>=0.15.0",
    "sphinx-autodoc-typehints>=1.12.0",
]

# Testing requirements
test_requires = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "pytest-mock>=3.6.0",
    "pytest-xdist>=2.3.0",
    "pytest-benchmark>=3.4.0",
]

# API requirements
api_requires = [
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "python-multipart>=0.0.5",
    "prometheus-client>=0.11.0",
    "redis>=3.5.0",
]

# All extra requirements
extras_require = {
    "dev": dev_requires + docs_requires + test_requires + api_requires,
    "docs": docs_requires,
    "test": test_requires,
    "api": api_requires,
    "all": dev_requires + docs_requires + test_requires + api_requires,
}

setup(
    name="uncertainty-ids",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Uncertainty-Aware Intrusion Detection System based on Bayesian Ensemble Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/uncertainty-ids",
    project_urls={
        "Bug Reports": "https://github.com/research-team/uncertainty-ids/issues",
        "Source": "https://github.com/research-team/uncertainty-ids",
        "Documentation": "https://uncertainty-ids.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "uncertainty-ids-train=uncertainty_ids.cli.train:main",
            "uncertainty-ids-evaluate=uncertainty_ids.cli.evaluate:main",
            "uncertainty-ids-serve=uncertainty_ids.cli.serve:main",
            "uncertainty-ids-preprocess=uncertainty_ids.cli.preprocess:main",
        ],
    },
    include_package_data=True,
    package_data={
        "uncertainty_ids": [
            "configs/*.yaml",
            "data/schemas/*.json",
            "api/templates/*.html",
        ],
    },
    zip_safe=False,
    keywords=[
        "intrusion detection",
        "uncertainty quantification",
        "bayesian deep learning",
        "transformers",
        "cybersecurity",
        "machine learning",
        "neural networks",
        "ensemble methods",
    ],
    # Additional metadata
    platforms=["any"],
    license="MIT",
    # Test suite
    test_suite="tests",
    tests_require=test_requires,
    # Command line interface
    scripts=[],
    # Data files
    data_files=[],
)

# Post-installation message
print("""
🎉 Uncertainty-Aware Intrusion Detection System installed successfully!

📚 Documentation: https://uncertainty-ids.readthedocs.io/
🐛 Issues: https://github.com/research-team/uncertainty-ids/issues
💬 Discussions: https://github.com/research-team/uncertainty-ids/discussions

Quick start:
  1. Import the library: from uncertainty_ids import BayesianEnsembleIDS
  2. Check out examples: https://github.com/research-team/uncertainty-ids/tree/main/examples
  3. Read the docs: https://uncertainty-ids.readthedocs.io/en/latest/quickstart.html

For development installation:
  pip install -e ".[dev]"

Happy detecting! 🔍🛡️
""")
