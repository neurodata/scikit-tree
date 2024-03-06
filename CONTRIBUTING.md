# Contributing

Thanks for considering contributing! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

**Submodule dependency on a fork of scikit-learn**
Due to the current state of scikit-learn's internal Cython code for trees, we have to instead leverage a maintained fork of scikit-learn at <https://github.com/neurodata/scikit-learn>, where specifically, the `submodulev3` branch is used to build and install this repo. We keep that fork well-maintained and up-to-date with respect to the main sklearn repo. The only difference is the refactoring of the `tree/` submodule. This fork is used internally under the namespace ``sktree._lib.sklearn``. It is necessary to use this fork for anything related to:

- `RandomForest*`
- `ExtraTrees*`
- or any importable items from the `tree/` submodule, whether it is a Cython or Python object

If you are developing for scikit-tree, we will always depend on the most up-to-date commit of `https://github.com/neurodata/scikit-learn/submodulev3` as a submodule within scikit-tee. This branch is consistently maintained for changes upstream that occur in the scikit-learn tree submodule. This ensures that our fork maintains consistency and robustness due to bug fixes and improvements upstream

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/neurodata/scikit-tree/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/neurodata/scikit-tree/issues).  Be sure to include a clear title
and description.  The description should include as much relevant information as possible.  The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see.  Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create an feature request:

- Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
it first on a GitHub issue.
- Check the documentation to make sure your feature does not already exist.
- Do [a quick search](https://github.com/neurodata/scikit-tree/issues) to see whether your feature has already been suggested.

When creating your request, please:

- Provide a clear title and description.
- Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
- Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/scikit-tree.git

    or

        git clone git@github.com:USERNAME/scikit-tree.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/scikit-tree.git, but doesn't know anything the *main* repo, [https://github.com/neurodata/scikit-tree.git](https://github.com/neurodata/scikit-tree). You can see this by running

        # Note you should be in the "scikit-tree" directory. If you're not
        # run "cd ./scikit-tree" to change directory into the repo
        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/scikit-tree.git (fetch)
        origin https://github.com/USERNAME/scikit-tree.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/neurodata/scikit-tree.git](https://github.com/neurodata/scikit-tree). To do this, run the following:

        git remote add upstream https://github.com/neurodata/scikit-tree.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/scikit-tree.git (fetch)
        origin https://github.com/USERNAME/scikit-tree.git (push)
        upstream https://github.com/neurodata/scikit-tree.git (fetch)
        upstream https://github.com/neurodata/scikit-tree.git (push)

    Finally, you'll need to create a Python 3 virtual environment suitable for working on this project. There a number of tools out there that making working with virtual environments easier.
    The most direct way is with the [`venv` module](https://docs.python.org/3.7/library/venv.html) in the standard library, but if you're new to Python or you don't already have a recent Python 3 version installed on your machine,
    we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    On Mac, for example, you can install Miniconda with [Homebrew](https://brew.sh/):

        brew install miniconda

    Then you can create and activate a new Python environment by running:

        conda create -n scikit-tree python=3.9
        conda activate scikit-tree

    Once your virtual environment is activated, you can install your local clone in "editable mode" with

        pip install .[build]
        pip install -e .

    The "editable mode" comes from the `-e` argument to `pip`, and essential just creates a symbolic link from the site-packages directory of your virtual environment to the source code in your local clone. That way any changes you make will be immediately reflected in your virtual environment.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/allenai/python-package-temlate.git](https://github.com/neurodata/scikit-tree), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Developing and testing your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) testing runs [a number of checks](https://github.com/neurodata/scikit-tree/actions) for each pull request on [GitHub Actions](https://github.com/features/actions). You can run most of these tests locally, which is something you should do *before* opening a PR to help speed up the review process and make it easier for us. Please see our [development guide](https://github.com/neurodata/scikit-tree/blob/main/DEVELOPING.md) for a comprehensive overview of useful commands. This will cover aspects of code style checking, unit testing, integration testing, and building the documentation.

    And finally, please update the [CHANGELOG](https://github.com/neurodata/scikit-tree/docs/whats_new.rst) with notes on your contribution in the "Unreleased" section at the top.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/neurodata/scikit-tree/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Installing locally with Meson

Meson is a modern build system with a lot of nice features, which is why we use it for our build system to compile the Cython/C++ code.
However, there are some intricacies that might be new to a pure Python developer.

In general, the steps to build scikit-tree are:

- install scikit-learn fork (the wheels are released on pypi, so it should be ideally trivial).
- build and install scikit-tree locally using `spin`

Example would be:

        pip uninstall scikit-learn

        # install the fork of scikit-learn
        # Note: make sure you do not have actual scikit-learn installed
        pip install scikit-learn-tree

        # build
        ./spin build -j 4

The above code assumes you have installed all the relevant build packages. See the repo's README for more info. A lot of the build packages might differ from OS to OS. We recommend following scikit-learn's documentation on building locally.

The most common errors come from the following:

1. clashing sklearn: if they have scikit-learn main installed for another project in the same virtual env, then they should start with a clean virtual env. There is unfortunately no easy way around this rn. Maybe if we vendor the scikit-learn-fork changes, we can get around this in the future.
2. not all necessary compile-time packages installed: e.g. cython, compilers, etc. See scikit-learn documentation on building and installing from source.
3. not using spin CLI: scikit-tree uses Meson for its build because that’s what numpy/scipy uses and it’s actually quite robust. However, it has a few limitations. We use spin CLI to run commands such as testing, shell that has sktree installed, etc.  This is because the Python PATH cannot differentiate between sktree/ in your repo directory vs the sktree/ it sees installed. So if you really want to run something in your teminal that is not using spin, then you have to change directories.
4. Not having certain computer packages installed, such as gcc, etc.: Scikit-tree requires compiling Cython and C++ code. Therefore, if you have outdated C/C++ compilers, prolly can be the issue.

The CI files for github actions shows how to build and install for each OS.

### Writing docstrings

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to build our API docs, which automatically parses all docstrings
of public classes and methods. All docstrings should adhere to the [Numpy styling convention](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).

### Testing Changes Locally

We have included a few convenience functions to check your code. These checks must pass and will be checked by the PR's continuous integration services. You can install the various different developer dependencies:

    pip install .[test,style,doc]

You can verify that your code will pass certain style, formatting and lint checks by running:

    make run-checks

### Documentation

If you need to build the documentation locally and check for doc errors:

    make build-doc


## Developing a new Tree model

Here, we define some high-level procedures for how to best approach implementing a new decision-tree model that is not supported yet in scikit-tree.

1. First-pass on implementation:

    Implement a Cython splitter class and expose it in Python afterwards. Follow the framework for PatchObliqueSplitter and ObliqueSplitter and their respective decision-tree models: PatchObliqueDecisionTreeClassifier and ObliqueDecisionTreeClassifier.

2. Second-pass on implementation:

    This involves extending relevant API beyond just the Splitter in Cython. This requires maintaining some degree of backwards-compatibility. Extend the existing API for Tree, TreeBuilder, Criterion, or ObliqueSplitter to enable whatever functionality you desire.

3. Third-pass on implementation:

    This is the most complex implementation and should in theory be rarely used.  This involves both designing a change in the scikit-learn fork submodule as well as relevant changes in scikit-tree itself. Extend the scikit-learn fork API. This requires maintaining some degree of backwards-compatability and testing the proposed changes wrt whatever changes you then make in scikit-tree.

---

The Project abides by the Organization's [code of conduct](https://github.com/py-why/governance/blob/main/CODE-OF-CONDUCT.md) and [trademark policy](https://github.com/py-why/governance/blob/main/TRADEMARKS.md).

---
Part of MVG-0.1-beta.
Made with love by GitHub. Licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
