"""
Setup script for Acoustic Simulation Framework
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="acoustic-sim-framework",
    version="1.0.0",
    author="Manus AI",
    author_email="dev@manus.ai",
    description="A highly configurable acoustic simulation framework for generating training data for in-vehicle noise source localization and separation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/acoustic-sim-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "acoustic-sim=scripts.generate_dataset:main",
        ],
    },
)
