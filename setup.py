from pathlib import Path

from setuptools import find_packages, setup

base_path = Path(__file__).parent
long_description = (base_path / "README.md").read_text()

setup(
    name='brown_clustering',
    packages=find_packages(include=("brown_clustering",)),
    version='0.1.2',
    description='Fast Brown Clustering',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='bfuchs',
    author_email="benedikt.fuchs.staw@hotmail.com",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "numpy<1.21",
        "numba",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mypy",
            "pytest-profiling",
            "pytest-flake8",
            "flake8<4.0.0",
            "flake8-isort",
        ]
    },
)
