:orphan:

Installation
============

Dependencies
------------

* ``numpy`` (>=1.23)
* ``scipy`` (>=1.5.0)
* ``scikit-learn`` (>=1.3)
* ``joblib`` (>=1.0.0)
* ``matplotlib`` (optional)

**scikit-tree** supports Python >= 3.9.

Installing with ``pip``
-----------------------

**scikit-tree** is available [on PyPI](https://pypi.org/project/scikit-tree/). Just run

.. code-block:: bash

    pip install sktree

Installing from source with Meson
---------------------------------

To install **scikit-tree** from source, first clone [the repository](https://github.com/neurodata/scikit-tree):

.. code-block:: bash

    git clone https://github.com/neurodata/scikit-tree.git
    cd scikit-tree

    # ideally, you should always start within a virtual environment
    conda create -n sklearn-dev python=3.9
    conda activate sklearn-dev

Then run installation of build packages

.. code-block:: bash

    pip install -r build_requirements.txt
    pip install spin

    # use spin CLI to run Meson build locally
    ./spin build -j 2

    # you can now run tests
    ./spin test

via pip, you will be able to install in editable mode (pending Meson-Python support).

.. code-block:: bash

    pip install -e .

    # if editing Cython files
    pip install --verbose --no-build-isolation --editable .

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/neurodata/scikit-tree/zipball/master

Conda (Recommended)
-------------------
First, create a virtual environment using Conda.

    conda create -n sklearn-dev python=3.9

# activate the virtual environment and install necessary packages to build from source

    conda activate sklearn-dev
    conda install -c conda-forge numpy scipy cython joblib threadpoolctl pytest compilers llvm-openmp poetry

Next, `sktree` from source:

    pip install -e .

    # if editing Cython files
    pip install --verbose --no-build-isolation --editable .

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # if you would like an editable install of dodiscover for dev purposes
    pip install -e .

    pip install https://api.github.com/repos/neurodata/scikit-tree/zipball/main


    pip install https://api.github.com/repos/neurodata/scikit-learn/zipball/obliquepr

Note that currently, we need to build the development version of scikit-learn with oblique trees within this [PR](https://github.com/scikit-learn/scikit-learn/pull/22754).

Checkout this PR code, and build from source, using scikit-learn's build from source page instructions.
