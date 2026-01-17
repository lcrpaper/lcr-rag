#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LCR (Learned Conflict Resolution) - Setup Script

This is a fallback installation script when pyproject.toml is not supported.
For modern Python (3.10+), prefer: pip install -e .
"""

import os
from pathlib import Path

from setuptools import find_packages, setup


VERSION = "1.0.0"


this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.35.0,<4.40.0",
    "tokenizers>=0.15.0",
    "datasets>=2.14.0",
    "accelerate>=0.24.0",
    "huggingface-hub>=0.19.0",
    "safetensors>=0.4.0",
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "pyyaml>=6.0.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    "python-dotenv>=1.0.0",
    "rank-bm25>=0.2.2",
    "tqdm>=4.66.0",
    "rich>=13.6.0",
    "click>=8.1.0",
    "pydantic>=2.5.0",
]


DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.4.0",
    "hypothesis>=6.91.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.6.0",
]

TEST_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.4.0",
    "hypothesis>=6.91.0",
]

GPU_REQUIRES = [
    "bitsandbytes>=0.41.0",
    "peft>=0.6.0",
    "optimum>=1.14.0",
]

TRACKING_REQUIRES = [
    "wandb>=0.16.0",
    "mlflow>=2.9.0",
    "tensorboard>=2.14.0",
]

VIZ_REQUIRES = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
]

NOTEBOOK_REQUIRES = [
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "ipywidgets>=8.1.0",
]

API_REQUIRES = [
    "openai>=1.3.0",
    "anthropic>=0.7.0",
    "cohere>=4.37.0",
]

INFERENCE_REQUIRES = [
    "vllm>=0.2.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.16.0",
]

DISTRIBUTED_REQUIRES = [
    "deepspeed>=0.12.0",
    "ray[default]>=2.8.0",
]

DOCS_REQUIRES = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",
]

FULL_REQUIRES = (
    DEV_REQUIRES
    + GPU_REQUIRES
    + TRACKING_REQUIRES
    + VIZ_REQUIRES
    + NOTEBOOK_REQUIRES
    + API_REQUIRES
    + DISTRIBUTED_REQUIRES
    + DOCS_REQUIRES
)

EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
    "test": TEST_REQUIRES,
    "gpu": GPU_REQUIRES,
    "tracking": TRACKING_REQUIRES,
    "viz": VIZ_REQUIRES,
    "notebook": NOTEBOOK_REQUIRES,
    "api": API_REQUIRES,
    "inference": INFERENCE_REQUIRES,
    "distributed": DISTRIBUTED_REQUIRES,
    "docs": DOCS_REQUIRES,
    "full": FULL_REQUIRES,
}


setup(
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "natural-language-processing",
        "machine-learning",
        "conflict-resolution",
        "retrieval-augmented-generation",
        "transformers",
        "deep-learning",
    ],
    packages=find_packages(
        where=".",
        include=["src*", "configs*"],
        exclude=["tests*", "_backup*", "sample*"],
    ),
    package_dir={"": "."},
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
        "configs": ["*.yaml", "**/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "lcr=src.models.lcr_system:main",
            "lcr-train=src.training.train_detector:main",
            "lcr-eval=scripts.evaluation.eval_main_results:main",
        ],
    },
    zip_safe=False,
)
