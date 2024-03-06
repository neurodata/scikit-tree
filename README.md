[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/neurodata/scikit-tree/tree/main.svg?style=svg)](https://circleci.com/gh/neurodata/scikit-tree/tree/main)
[![Main](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/neurodata/scikit-tree/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/neurodata/scikit-tree/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/neurodata/scikit-tree)
[![PyPI Download count](https://img.shields.io/pypi/dm/scikit-tree.svg)](https://pypistats.org/packages/scikit-tree)
[![Latest PyPI release](https://img.shields.io/pypi/v/scikit-tree.svg)](https://pypi.org/project/scikit-tree/)
[![DOI](https://zenodo.org/badge/491260497.svg)](https://zenodo.org/doi/10.5281/zenodo.8412279)

scikit-tree
===========

scikit-tree is a scikit-learn compatible API for building state-of-the-art decision trees. These include unsupervised trees, oblique trees, uncertainty trees, quantile trees and causal trees.

Tree-models have withstood the test of time, and are consistently used for modern-day data science and machine learning applications. They especially perform well when there are limited samples for a problem and are flexible learners that can be applied to a wide variety of different settings, such as tabular, images, time-series, genomics, EEG data and more.

Documentation
=============

See here for the documentation for our dev version: <https://docs.neurodata.io/scikit-tree/dev/index.html>

Why oblique trees and why trees beyond those in scikit-learn?
=============================================================

In 2001, Leo Breiman proposed two types of Random Forests. One was known as ``Forest-RI``, which is the axis-aligned traditional random forest. One was known as ``Forest-RC``, which is the random oblique linear combinations random forest. This leveraged random combinations of features to perform splits. [MORF](1) builds upon ``Forest-RC`` by proposing additional functions to combine features. Other modern tree variants such as Canonical Correlation Forests (CCF), Extended Isolation Forests, Quantile Forests, or unsupervised random forests are also important at solving real-world problems using robust decision tree models.

Installation
============

Our installation will try to follow scikit-learn installation as close as possible, as we contain Cython code subclassed, or inspired by the scikit-learn tree submodule.

Dependencies
------------

We minimally require:

    * Python (>=3.9)
    * numpy
    * scipy
    * scikit-learn >= 1.3

Installation with Pip (<https://pypi.org/project/scikit-tree/>)
-------------------------------------------------------------

Installing with pip on a conda environment is the recommended route.

    pip install scikit-tree

Building locally with Meson (For developers)
--------------------------------------------

Make sure you have the necessary packages installed

    # install build dependencies
    pip install .[build]

    # you may need these optional dependencies to build scikit-learn locally
    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

We use the ``spin`` CLI to abstract away build details:

    # run the build using Meson/Ninja
    ./spin build

    # you can run the following command to see what other options there are
    ./spin --help
    ./spin build --help

    # For example, you might want to start from a clean build
    ./spin build --clean

    # or build in parallel for faster builds
    ./spin build -j 2

    # you will need to double check the build-install has the proper path
    # this might be different from machine to machine
    export PYTHONPATH=${PWD}/build-install/usr/lib/python3.9/site-packages

    # run specific unit tests
    ./spin test -- sktree/tree/tests/test_tree.py

    # you can bring up the CLI menu
    ./spin --help

You can also do the same thing using Meson/Ninja itself. Run the following to build the local files:

    # generate ninja make files
    meson build --prefix=$PWD/build

    # compile
    ninja -C build

    # install scikit-tree package
    meson install -C build

    export PYTHONPATH=${PWD}/build/lib/python3.9/site-packages

    # to check installation, you need to be in a different directory
    cd docs;  
    python -c "from sktree import tree"
    python -c "import sklearn; print(sklearn.__version__);"

After building locally, you can use editable installs (warning: this only registers Python changes locally)

    pip install --no-build-isolation --editable .

Or if you have spin v0.8+ installed, you can just run directly

    spin install

Development
===========

We welcome contributions for modern tree-based algorithms. We use Cython to achieve fast C/C++ speeds, while abiding by a scikit-learn compatible (tested) API. Moreover, our Cython internals are easily extensible because they follow the internal Cython API of scikit-learn as well.

Due to the current state of scikit-learn's internal Cython code for trees, we have to instead leverage a fork of scikit-learn at <https://github.com/neurodata/scikit-learn> when
extending the decision tree model API of scikit-learn. Specifically, we extend the Python and Cython API of the tree submodule in scikit-learn in our submodule, so we can introduce the tree models housed in this package. Thus these extend the functionality of decision-tree based models in a way that is not possible yet in scikit-learn itself. As one example, we introduce an abstract API to allow users to implement their own oblique splits. Our plan in the future is to benchmark these functionalities and introduce them upstream to scikit-learn where applicable and inclusion criterion are met.

References
==========

[1]: [`Li, Adam, et al. "Manifold Oblique Random Forests: Towards Closing the Gap on Convolutional Deep Networks" SIAM Journal on Mathematics of Data Science, 5(1), 77-96, 2023`](https://doi.org/10.1137/21M1449117)
