# The major setup script has been moved to pyproject.toml. This file is kept for compatibility and will be deprecated in the future.

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.21",
    "sympy>=1.9",
    "scipy>=1.7",
    "cvxpy>=1.2",
    "juliacall>=0.9.35",
    "more_itertools>=8.12",
    "h5py>=3.6",
]

test_requirements = [
    "coverage>=4.5.4",
]

setup(
    author="Jing Liu et al.",
    author_email="jing.liu@hainanu.edu.cn ",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="A package for quantum parameter estimation.",
    install_requires=requirements,
    license="BSD 3-Clause License",
    long_description=readme + "\n\n" + history,
    long_description_content_type = "text/markdown", 
    include_package_data=True,
    keywords="quanestimation",
    name="quanestimation",
    packages=find_packages(include=["quanestimation", "quanestimation.*"]),
    test_suite="test",
    tests_require=test_requirements,
    url="https://github.com/QuanEstimation/QuanEstimation",
    version="0.4.0",
    zip_safe=False,
)
