# The major setup script has been moved to pyproject.toml. This file is kept for compatibility and will be deprecated in the future.

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
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
    "scipy",
    "cvxpy",
    "juliacall",
    "more_itertools",
    "julia_project",
    "h5py",
]

test_requirements = []

setup(
    author="Jing Liu et al.",
    author_email="liujing@hainanu.edu.cn ",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
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
    entry_points={
        "console_scripts": [
            "quanestimation=quanestimation.cli:main",
        ],
    },
    install_requires=requirements,
    license="BSD 3-Clause License",
    long_description=readme + "\n\n" + history,
    long_description_content_type = "text/markdown", 
    include_package_data=True,
    keywords="quanestimation",
    name="quanestimation",
    packages=find_packages(include=["quanestimation", "quanestimation.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/QuanEstimation/QuanEstimation",
    version="0.2.8",
    zip_safe=False,
)
