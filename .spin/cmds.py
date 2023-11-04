import os
import subprocess
import sys

import click
from spin import util
from spin.cmds import meson


def get_git_revision_hash(submodule) -> str:
    return subprocess.check_output(["git", "rev-parse", f"@:{submodule}"]).decode("ascii").strip()


@click.command()
@click.argument("slowtest", default=True)
@click.pass_context
def coverage(ctx, slowtest):
    """📊 Generate coverage report"""
    if slowtest:
        pytest_args = (
            "-o",
            "python_functions=test_*",
            "sktree",
            "--cov=sktree",
            "--cov-report=xml",
            "-k .",
        )
    else:
        pytest_args = (
            "-o",
            "python_functions=test_*",
            "sktree",
            "--cov=sktree",
            "--cov-report=xml",
        )
    ctx.invoke(meson.test, pytest_args=pytest_args)


@click.command()
@click.option("--forcesubmodule", is_flag=False, help="Force submodule pull.")
def setup_submodule(forcesubmodule=False):
    """Build scikit-tree using submodules.

    git submodule set-branch -b submodulev2 sktree/_lib/sklearn

    git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    commit_fpath = "./sktree/_lib/commit.txt"
    submodule = "./sktree/_lib/sklearn_fork"
    commit = ""
    current_hash = ""

    # if the forked folder does not exist, we will need to force update the submodule
    if not os.path.exists("./sktree/_lib/sklearn/") or forcesubmodule:
        # update git submodule
        util.run(["git", "submodule", "update", "--init", "--force"])
    else:
        # update git submodule
        util.run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--force",
            ]
        )

    # get the commit hash if the commmit file exists
    if os.path.exists(commit_fpath):
        with open(commit_fpath, "r") as f:
            commit = f.read().strip()

    # get revision hash
    current_hash = get_git_revision_hash(submodule)

    print(current_hash)
    print(commit)

    # if the commit file doesn't exist or the commit hash is different, we need
    # to update our sklearn repository
    if current_hash == "" or current_hash != commit:
        util.run(
            [
                "mkdir",
                "-p",
                "./sktree/_lib/sklearn/",
            ],
        )
        util.run(
            [
                "touch",
                commit_fpath,
            ],
        )
        print(commit_fpath)
        with open(commit_fpath, "w") as f:
            f.write(current_hash)
        print(commit, current_hash)

        util.run(
            [
                "rm",
                "-rf",
                "sktree/_lib/sklearn",
            ]
        )

        if os.path.exists("sktree/_lib/sklearn_fork/sklearn") and (commit != current_hash):
            util.run(
                [
                    "cp",
                    "-r",
                    "sktree/_lib/sklearn_fork/sklearn",
                    "sktree/_lib/sklearn",
                ]
            )


@click.command()
@click.option("-j", "--jobs", help="Number of parallel tasks to launch", type=int)
@click.option("--clean", is_flag=True, help="Clean build directory before build")
@click.option(
    "--forcesubmodule", is_flag=True, help="Force submodule pull.", envvar="FORCE_SUBMODULE"
)
@click.option("-v", "--verbose", is_flag=True, help="Print all build output, even installation")
@click.argument("meson_args", nargs=-1)
@click.pass_context
def build(ctx, meson_args, jobs=None, clean=False, forcesubmodule=False, verbose=False):
    """Build scikit-tree using submodules.

    git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    ctx.invoke(setup_submodule, forcesubmodule=forcesubmodule)

    # run build as normal
    ctx.invoke(meson.build, meson_args=meson_args, jobs=jobs, clean=clean, verbose=verbose)


@click.command()
@click.argument("asv_args", nargs=-1)
def asv(asv_args):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- dev -b TransformSuite

    ./spin asv -- continuous --verbose --split --bench ObliqueRandomForest origin/main constantsv2

    Please see CONTRIBUTING.txt
    """
    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-tree found; run `spin build` first.")
        sys.exit(1)

    os.environ["ASV_ENV_DIR"] = "/Users/adam2392/miniforge3"
    os.environ["PYTHONPATH"] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(["asv"] + list(asv_args))
