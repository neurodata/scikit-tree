import os
import shutil
import sys

import click
from devpy import util
from devpy.cmds.meson import get_site_packages


@click.command()
@click.option("--build-dir", default="build", help="Build directory; default is `$PWD/build`")
@click.option("--clean", is_flag=True, help="Clean previously built docs before building")
def docs(build_dir, clean=False):
    """📖 Build documentation"""
    if clean:
        doc_dir = "./docs/_build"
        if os.path.isdir(doc_dir):
            print(f"Removing `{doc_dir}`")
            shutil.rmtree(doc_dir)

    site_path = get_site_packages()
    if site_path is None:
        print("No built scikit-tree found; run `./dev.py build` first.")
        sys.exit(1)

    util.run(["pip", "install", "-q", "-r", "doc_requirements.txt"])

    os.environ["SPHINXOPTS"] = "-W"
    os.environ["PYTHONPATH"] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(["make", "-C", "docs", "html"], replace=True)


@click.command()
def coverage():
    """📊 Generate coverage report"""
    util.run(
        [
            "python",
            "-m",
            "devpy",
            "test",
            "--",
            "-o",
            "python_functions=test_*",
            "sktree",
            "--cov=sktree",
            "--cov-report=xml",
        ],
        replace=True,
    )
