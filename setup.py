#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pip>=19.2.3",
    "bump2version>=0.5.11",
    "wheel>=0.33.6",
    "coverage>=4.5.4",
    "numpy",
    "sympy",
    "matplotlib",
    "jill",
    "scipy",
    "qutip",
    "cvxpy",
    "julia",
]

test_requirements = []

setup(
    author="Huaiming Yu",
    author_email="huaimingyuuu@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A package for quantum estimation.",
    entry_points={
        "console_scripts": [
            "quanestimation=quanestimation.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="quanestimation",
    name="quanestimation",
    packages=find_packages(include=["quanestimation", "quanestimation.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/QuanEstimation/QuanEstimation",
    version="0.1.0",
    zip_safe=False,
)
