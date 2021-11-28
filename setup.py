from setuptools import find_packages, setup

setup(
    name='brown_clustering',
    packages=find_packages(include=("brown_clustering",)),
    version='0.1.0',
    description='Fast Brown Clustering',
    author='bfuchs',
    author_email="benedikt.fuchs.staw@hotmail.com",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "numba",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mypy",
            "pytest-flake8",
            "flake8<4.0.0",
            "flake8-isort",
        ]
    },
)
