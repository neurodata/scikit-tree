# Requirements
* Python 3.8+
* Poetry (`curl -sSL https://install.python-poetry.org | python - --version=1.2.2`)

For the other requirements, inspect the ``pyproject.toml`` file. If you are updated the dependencies, please run `poetry update` to update the

# Development Tasks
There are a series of top-level tasks available through Poetry. These can each be run via

 `poetry run poe <taskname>`

### Basic Verification
* **format** - runs the suite of formatting tools applying tools to make code compliant
* **format_check** - runs the suite of formatting tools checking for compliance
* **lint** - runs the suite of linting tools
* **type_check** - performs static typechecking of the codebase using mypy
* **unit_test** - executes fast unit tests
* **verify** - executes the basic PR verification suite, which includes all the tasks listed above

### Docsite
* **build_docs** - build the API documentation site
* **build_docs_noplot** - build the API documentation site without running explicitly any of the examples, for faster local checks of any documentation updates.

## Details

Here we provide some details to understand the development process.

### Coding Style

For convenience ``poetry`` provides a command line interface for running all the necessary development commands:

    poetry run poe format

This will run isort and black on the entire repository. This will auto-format the code to comply with our coding style.

### Lint

We use linting services to check for common errors in the code.

    poetry run poe lint

We use flake8, bandit, codespell and pydocstyle to check for code smells, which are lines of code that can lead to unintended errors.

### Type checking

We use type checking to check for possible runtime errors due to mismatched types. Python is dynamically typed, so this helps us and the user catch errors that would otherwise then occur during runtime. We use mypy to perform type checking.

    poetry run poe type_check

### Unit tests

In order for any code to be added to the repository, we require unit tests to pass. Any new code should be accompanied by unit tests.

    poetry run poe unit_test

# Cython and C++
The general design of scikit-tree follows that of the tree-models inside scikit-learn, where tree-based models are inherently Cythonized, or written with C++. Then the actual forest (e.g. RandomForest, or ExtraForest) is just a Python API wrapper that creates an ensemble of the trees.

In order to develop new tree models, generally Cython and C++ code will need to be written in order to optimize the tree building process, otherwise fitting a single forest model would take very long.

# Making a Release

Scikit-tree is in-line with scikit-learn and thus relies on each new version released there. Moreover, scikit-tree relies on compiled code, so releases are a bit more complex than the typical Python package.

1. Download wheels from GH Actions and put all wheels into a ``dist/`` folder

https://github.com/neurodata/scikit-tree/actions/workflows/build_wheels.yml will have all the wheels for common OSes built for each Python version.

2. Upload wheels to test PyPi

    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Verify that installations work as expected on your machine.

3. Upload wheels

    twine upload dist/*

4. Update version number on ``meson.build`` and ``_version.py`` to the relevant version.


# Submodule

To update the submodule's branch for development purposes, one can run:

    git submodule set-branch -b <branch_name> sktree/_lib/sklearn