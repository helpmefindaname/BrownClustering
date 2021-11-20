from setuptools import find_packages, setup
setup(
    name='brown_clustering',
    packages=find_packages(include=("brown_clustering",)),
    version='0.1.0',
    description='Fast Brown Clustering',
    author='bfuchs',
    license='',
    install_requires=[
        "nltk",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mypy",
            "pytest-flake8",
            "flake8<4.0.0",
            "flake8-import-order",
            "flake8-cognitive-complexity",
        ]
    },
)
