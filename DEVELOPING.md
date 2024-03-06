<!-- TOC -->

- [Requirements](#requirements)
- [Setting up your development environment](#setting-up-your-development-environment)
- [Building the project from source](#building-the-project-from-source)
- [Development Tasks](#development-tasks)
- [Advanced Updating submodules](#advanced-updating-submodules)
- [Cython and C++](#cython-and-c)
- [Making a Release](#making-a-release)

<!-- /TOC -->

# Requirements

- Python 3.9+
- numpy>=1.25
- scipy>=1.11
- scikit-learn>=1.3.1

For the other requirements, inspect the ``pyproject.toml`` file.

# Setting up your development environment

We recommend using miniconda, as python virtual environments may not setup properly compilers necessary for our compiled code. For detailed information on setting up and managing conda environments, see <https://conda.io/docs/test-drive.html>.

<!-- Setup a conda env -->

    conda create -n sktree
    conda activate sktree

**Make sure you specify a Python version if your system defaults to anything less than Python 3.9.**

**Any commands should ALWAYS be after you have activated your conda environment.**
Next, install necessary build dependencies. For more information, see <https://scikit-learn.org/stable/developers/advanced_installation.html>.

    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

Assuming these steps have worked properly and you have read and followed any necessary scikit-learn advanced installation instructions, you can then install dependencies for scikit-tree.

If you are developing locally, you will need the build dependencies to compile the Cython / C++ code:

    pip install .[build]

Other requirements can be installed as such:

    pip install .
    pip install .[style]
    pip install .[test]
    pip install .[doc]

# Building the project from source

We leverage meson to build scikit-tree from source. We utilize a CLI tool, called [spin](https://github.com/scientific-python/spin), which wraps certain meson commands to make building easier.

For example, the following command will build the project completely from scratch

    spin build --clean

If you have part of the build already done, you can run:

    spin build

The following command will test the project

    spin test

For other commands, see

    spin --help

Note at this stage, you will be unable to run Python commands directly. For example, ``pytest ./sktree`` will not work.

However, after installing and building the project from source using meson, you can leverage editable installs to make testing code changes much faster. For more information on meson-python's progress supporting editable installs in a better fashion, see <https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html>.

    pip install --no-build-isolation --editable .

**Note: editable installs for scikit-tree REQUIRE you to have built the project using meson already.** This will now link the meson build to your Python runtime. Now if you run

    pytest ./sktree

the unit-tests should run.

# Development Tasks

There are a series of top-level tasks available.

    make run-checks

This leverage pre-commit to run a series of precommit checks.

# (Advanced) Updating submodules

Scikit-tree relies on a submodule of a forked-version of scikit-learn for certain Python and Cython code that extends the ``DecisionTree*`` models. Usually, if a developer is making changes, they should go over to the ``submodulev3`` branch on ``https://github.com/neurodata/scikit-learn`` and
submit a PR to make changes to the submodule.

This should **ALWAYS** be supported by some use-case in scikit-tree. We want the minimal amount of code-change in our forked version of scikit-learn to make it very easy to merge in upstream changes, bug fixes and features for tree-based code.

Once a PR is submitted and merged, the developer can update the submodule here in scikit-tree, so that we leverage the new commit. You **must** update the submodule commit ID and also commit this change, so that way the build leverages the new submodule commit ID.

    git submodule update --init --recursive --remote
    git add -A
    git commit -m "Update submodule" -s

Now, you can re-build the project using the latest submodule changes.

    spin build --clean

# Cython and C++

The general design of scikit-tree follows that of the tree-models inside scikit-learn, where tree-based models are inherently Cythonized, or written with C++. Then the actual forest (e.g. RandomForest, or ExtraForest) is just a Python API wrapper that creates an ensemble of the trees.

In order to develop new tree models, generally Cython and C++ code will need to be written in order to optimize the tree building process, otherwise fitting a single forest model would take very long.

# Making a Release

Scikit-tree is in-line with scikit-learn and thus relies on each new version released there. Moreover, scikit-tree relies on compiled code, so releases are a bit more complex than the typical Python package.

1. Download wheels from GH Actions and put all wheels into a ``dist/`` folder

<https://github.com/neurodata/scikit-tree/actions/workflows/build_wheels.yml> will have all the wheels for common OSes built for each Python version.

2. Upload wheels to test PyPi

```
twine upload dist/* --repository testpypi
```

Verify that installations work as expected on your machine.

3. Upload wheels

```
twine upload dist/*
```

or if you have two-factor authentication enabled: <https://pypi.org/help/#apitoken>

    twine upload dist/* --repository scikit-tree

4. Update version number on ``meson.build`` and ``pyproject.toml`` to the relevant version.

See https://github.com/neurodata/scikit-tree/pull/160 as an example.
