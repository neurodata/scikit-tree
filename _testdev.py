#! /usr/bin/env python3

'''
See: https://github.com/scipy/scipy/blob/main/.circleci/config.yml

Developer CLI: building (meson), tests, benchmark, etc.

This file contains tasks definitions for doit (https://pydoit.org).
And also a CLI interface using click (https://click.palletsprojects.com).

The CLI is ideal for project contributors while,
doit interface is better suited for authoring the development tasks.

REQUIREMENTS:
--------------
- see environment.yml: doit, pydevtool, click, rich-click

# USAGE:

## 1 - click API

Commands can added using default Click API. i.e.

```
@cli.command()
@click.argument('extra_argv', nargs=-1)
@click.pass_obj
def python(ctx_obj, extra_argv):
    """Start a Python shell with PYTHONPATH set"""
```

## 2 - class based Click command definition

`CliGroup` provides an alternative class based API to create Click commands.

Just use the `cls_cmd` decorator. And define a `run()` method

```
@cli.cls_cmd('test')
class Test():
    """Run tests"""

    @classmethod
    def run(cls):
        print('Running tests...')
```

- Command may make use a Click.Group context defining a `ctx` class attribute
- Command options are also define as class attributes

```
@cli.cls_cmd('test')
class Test():
    """Run tests"""
    ctx = CONTEXT

    verbose = Option(
        ['--verbose', '-v'], default=False, is_flag=True, help="verbosity")

    @classmethod
    def run(cls, **kwargs): # kwargs contains options from class and CONTEXT
        print('Running tests...')
```

## 3 - class based interface can be run as a doit task by subclassing from Task

- Extra doit task metadata can be defined as class attribute TASK_META.
- `run()` method will be used as python-action by task

```
@cli.cls_cmd('test')
class Test(Task):   # Task base class, doit will create a task
    """Run tests"""
    ctx = CONTEXT

    TASK_META = {
        'task_dep': ['build'],
    }

    @classmethod
    def run(cls, **kwargs):
        pass
```

## 4 - doit tasks with cmd-action "shell" or dynamic metadata

Define method `task_meta()` instead of `run()`:

```
@cli.cls_cmd('refguide-check')
class RefguideCheck(Task):
    @classmethod
    def task_meta(cls, **kwargs):
        return {
```

'''

import contextlib
import datetime
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from sysconfig import get_path

# distutils is required to infer meson install path
# if this needs to be replaced for Python 3.12 support and there's no
# stdlib alternative, use CmdAction and the hack discussed in gh-16058
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from distutils import dist
    from distutils.command.install import INSTALL_SCHEMES

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType as new_module

import click
from click import Argument, Option
from doit.api import run_tasks
from doit.cmd_base import ModuleTaskLoader
from doit.exceptions import TaskError
from doit.reporter import ZeroReporter
from pydevtool.cli import CliGroup, Task, UnifiedContext
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich_click import rich_click

DOIT_CONFIG = {
    "verbosity": 2,
    "minversion": "0.36.0",
}


console_theme = Theme(
    {
        "cmd": "italic gray50",
    }
)


class EMOJI:
    cmd = ":computer:"


rich_click.STYLE_ERRORS_SUGGESTION = "yellow italic"
rich_click.SHOW_ARGUMENTS = True
rich_click.GROUP_ARGUMENTS_OPTIONS = False
rich_click.SHOW_METAVARS_COLUMN = True
rich_click.USE_MARKDOWN = True
rich_click.OPTION_GROUPS = {
    "dev.py": [
        {
            "name": "Options",
            "options": ["--help", "--build-dir", "--no-build", "--install-prefix"],
        },
    ],
    "dev.py test": [
        {
            "name": "Options",
            "options": ["--help", "--verbose", "--parallel", "--coverage", "--durations"],
        },
        {
            "name": "Options: test selection",
            "options": ["--submodule", "--tests", "--mode"],
        },
    ],
}
rich_click.COMMAND_GROUPS = {
    "dev.py": [
        {
            "name": "build & testing",
            "commands": ["build", "test"],
        },
        {
            "name": "static checkers",
            "commands": ["lint", "mypy"],
        },
        {
            "name": "environments",
            "commands": ["shell", "python", "ipython"],
        },
        {
            "name": "documentation",
            "commands": ["doc", "refguide-check"],
        },
        {
            "name": "release",
            "commands": ["notes", "authors"],
        },
        {
            "name": "benchmarking",
            "commands": ["bench"],
        },
    ]
}


class ErrorOnlyReporter(ZeroReporter):
    desc = """Report errors only"""

    def runtime_error(self, msg):
        console = Console()
        console.print("[red bold] msg")

    def add_failure(self, task, fail_info):
        console = Console()
        if isinstance(fail_info, TaskError):
            console.print(f"[red]Task Error - {task.name}" f" => {fail_info.message}")
        if fail_info.traceback:
            console.print(
                Panel(
                    "".join(fail_info.traceback),
                    title=f"{task.name}",
                    subtitle=fail_info.message,
                    border_style="red",
                )
            )


CONTEXT = UnifiedContext(
    {
        "build_dir": Option(
            ["--build-dir"],
            metavar="BUILD_DIR",
            default="build",
            show_default=True,
            help=":wrench: Relative path to the build directory.",
        ),
        "no_build": Option(
            ["--no-build", "-n"],
            default=False,
            is_flag=True,
            help=(
                ":wrench: Do not build the project"
                " (note event python only modification require build)."
            ),
        ),
        "install_prefix": Option(
            ["--install-prefix"],
            default=None,
            metavar="INSTALL_DIR",
            help=(
                ":wrench: Relative path to the install directory."
                " Default is <build-dir>-install."
            ),
        ),
    }
)


def run_doit_task(tasks):
    """
    :param tasks: (dict) task_name -> {options}
    """
    loader = ModuleTaskLoader(globals())
    doit_config = {
        "verbosity": 2,
        "reporter": ErrorOnlyReporter,
    }
    return run_tasks(loader, tasks, extra_config={"GLOBAL": doit_config})


class CLI(CliGroup):
    context = CONTEXT
    run_doit_task = run_doit_task


@click.group(cls=CLI)
@click.pass_context
def cli(ctx, **kwargs):
    """Developer Tool for scikit-tree

    \bCommands that require a built/installed instance are marked with :wrench:.


    \b**python dev.py --build-dir my-build test -s stats**

    """  # noqa: E501
    CLI.update_context(ctx, kwargs)


PROJECT_MODULE = "sktree"
PROJECT_ROOT_FILES = ["sktree", "LICENSE.txt", "meson.build"]


@dataclass
class Dirs:
    """
    root:
        Directory where scr, build config and tools are located
        (and this file)
    build:
        Directory where build output files (i.e. *.o) are saved
    install:
        Directory where .so from build and .py from src are put together.
    site:
        Directory where the built sktree version was installed.
        This is a custom prefix, followed by a relative path matching
        the one the system would use for the site-packages of the active
        Python interpreter.
    """

    # all paths are absolute
    root: Path
    build: Path
    installed: Path
    site: Path  # <install>/lib/python<version>/site-packages

    def __init__(self, args=None):
        """:params args: object like Context(build_dir, install_prefix)"""
        self.root = Path(__file__).parent.absolute()
        if not args:
            return
        self.build = Path(args.build_dir).resolve()
        if args.install_prefix:
            self.installed = Path(args.install_prefix).resolve()
        else:
            self.installed = self.build.parent / (self.build.stem + "-install")
        # relative path for site-package with py version
        # i.e. 'lib/python3.10/site-packages'
        self.site = self.get_site_packages()

    def add_sys_path(self):
        """Add site dir to sys.path / PYTHONPATH"""
        site_dir = str(self.site)
        sys.path.insert(0, site_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join((site_dir, os.environ.get("PYTHONPATH", "")))

    def get_site_packages(self):
        """
        Depending on whether we have debian python or not,
        return dist_packages path or site_packages path.
        """
        if "deb_system" in INSTALL_SCHEMES:
            # debian patched python in use
            install_cmd = dist.Distribution().get_command_obj("install")
            install_cmd.select_scheme("deb_system")
            install_cmd.finalize_options()
            plat_path = Path(install_cmd.install_platlib)
        else:
            plat_path = Path(get_path("platlib"))
        return self.installed / plat_path.relative_to(sys.exec_prefix)


@contextlib.contextmanager
def working_dir(new_dir):
    current_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(current_dir)


def import_module_from_path(mod_name, mod_path):
    """Import module with name `mod_name` from file path `mod_path`"""
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_test_runner(project_module):
    """
    get Test Runner from locally installed/built project
    """
    __import__(project_module)
    # sktree._lib._testutils:PytestTester
    test = sys.modules[project_module].test
    version = sys.modules[project_module].__version__
    mod_path = sys.modules[project_module].__file__
    mod_path = os.path.abspath(os.path.join(os.path.dirname(mod_path)))
    return test, version, mod_path


############


@cli.cls_cmd("build")
class Build(Task):
    """:wrench: Build & install package on path.

    \b
    ```python
    Examples:

    $ python dev.py build --asan ;
        ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true
        LD_PRELOAD=$(gcc --print-file-name=libasan.so)
        python dev.py test -v -t
        ./sktree/ndimage/tests/test_morphology.py -- -s
    ```
    """

    ctx = CONTEXT

    werror = Option(["--werror"], default=False, is_flag=True, help="Treat warnings as errors")
    gcov = Option(
        ["--gcov"],
        default=False,
        is_flag=True,
        help="enable C code coverage via gcov (requires GCC)." "gcov output goes to build/**/*.gc*",
    )
    debug = Option(["--debug", "-d"], default=False, is_flag=True, help="Debug build")
    parallel = Option(
        ["--parallel", "-j"],
        default=None,
        metavar="N_JOBS",
        help=("Number of parallel jobs for building. " "This defaults to 2 * n_cpus + 2."),
    )
    show_build_log = Option(
        ["--show-build-log"],
        default=False,
        is_flag=True,
        help="Show build output rather than using a log file",
    )

    @classmethod
    def setup_build(cls, dirs, args):
        """
        Setting up meson-build
        """
        for fn in PROJECT_ROOT_FILES:
            if not (dirs.root / fn).exists():
                print("To build the project, run dev.py in " "git checkout or unpacked source")
                sys.exit(1)

        env = dict(os.environ)
        cmd = ["meson", "setup", dirs.build, "--prefix", dirs.installed]
        build_dir = dirs.build
        run_dir = Path()
        if build_dir.exists() and not (build_dir / "meson-info").exists():
            if list(build_dir.iterdir()):
                raise RuntimeError(
                    "Can't build into non-empty directory " f"'{build_dir.absolute()}'"
                )

        build_options_file = build_dir / "meson-info" / "intro-buildoptions.json"
        if build_options_file.exists():
            with open(build_options_file) as f:
                build_options = json.load(f)
            installdir = None
            for option in build_options:
                if option["name"] == "prefix":
                    installdir = option["value"]
                    break
            if installdir != str(dirs.installed):
                run_dir = build_dir
                cmd = ["meson", "setup", "--reconfigure", "--prefix", str(dirs.installed)]
            else:
                return
        if args.werror:
            cmd += ["--werror"]
        if args.gcov:
            cmd += ["-Db_coverage=true"]
        # Setting up meson build
        cmd_str = " ".join([str(p) for p in cmd])
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")
        ret = subprocess.call(cmd, env=env, cwd=run_dir)
        if ret == 0:
            print("Meson build setup OK")
        else:
            print("Meson build setup failed!")
            sys.exit(1)
        return env

    @classmethod
    def build_project(cls, dirs, args, env):
        """
        Build a dev version of the project.
        """
        cmd = ["ninja", "-C", str(dirs.build)]
        if args.parallel is not None:
            cmd += ["-j", str(args.parallel)]

        # Building with ninja-backend
        cmd_str = " ".join([str(p) for p in cmd])
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")
        ret = subprocess.call(cmd, env=env, cwd=dirs.root)

        if ret == 0:
            print("Build OK")
        else:
            print("Build failed!")
            sys.exit(1)

    @classmethod
    def install_project(cls, dirs, args):
        """
        Installs the project after building.
        """
        if dirs.installed.exists():
            non_empty = len(os.listdir(dirs.installed))
            if non_empty and not dirs.site.exists():
                raise RuntimeError("Can't install in non-empty directory: " f"'{dirs.installed}'")
        cmd = ["meson", "install", "-C", args.build_dir, "--only-changed"]
        log_filename = dirs.root / "meson-install.log"
        start_time = datetime.datetime.now()
        cmd_str = " ".join([str(p) for p in cmd])
        cls.console.print(f"{EMOJI.cmd} [cmd] {cmd_str}")
        if args.show_build_log:
            ret = subprocess.call(cmd, cwd=dirs.root)
        else:
            print("Installing, see meson-install.log...")
            with open(log_filename, "w") as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=dirs.root)

            try:
                # Wait for it to finish, and print something to indicate the
                # process is alive, but only if the log file has grown (to
                # allow continuous integration environments kill a hanging
                # process accurately if it produces no output)
                last_blip = time.time()
                last_log_size = os.stat(log_filename).st_size
                while p.poll() is None:
                    time.sleep(0.5)
                    if time.time() - last_blip > 60:
                        log_size = os.stat(log_filename).st_size
                        if log_size > last_log_size:
                            elapsed = datetime.datetime.now() - start_time
                            print(
                                "    ... installation in progress ({} " "elapsed)".format(elapsed)
                            )
                            last_blip = time.time()
                            last_log_size = log_size

                ret = p.wait()
            except:  # noqa: E722
                p.terminate()
                raise
        elapsed = datetime.datetime.now() - start_time

        if ret != 0:
            if not args.show_build_log:
                with open(log_filename) as f:
                    print(f.read())
            print(f"Installation failed! ({elapsed} elapsed)")
            sys.exit(1)

        # ignore everything in the install directory.
        with open(dirs.installed / ".gitignore", "w") as f:
            f.write("*")

        print("Installation OK")
        return

    @classmethod
    def run(cls, add_path=False, **kwargs):
        kwargs.update(cls.ctx.get(kwargs))
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        args = Args(**kwargs)

        cls.console = Console(theme=console_theme)
        dirs = Dirs(args)
        if args.no_build:
            print("Skipping build")
        else:
            env = cls.setup_build(dirs, args)
            cls.build_project(dirs, args, env)
            cls.install_project(dirs, args)
            if args.win_cp_openblas and platform.system() == "Windows":
                if cls.copy_openblas(dirs) == 0:
                    print("OpenBLAS copied")
                else:
                    print("OpenBLAS copy failed!")
                    sys.exit(1)

        # add site to sys.path
        if add_path:
            dirs.add_sys_path()


@cli.cls_cmd("test")
class Test(Task):
    """:wrench: Run tests.

    \b
    ```python
    Examples:

    $ python dev.py test -s {SAMPLE_SUBMODULE}
    $ python dev.py test -t sktree.tree.tests.test_tree
    $ python dev.py test -s tree -m full --durations 20
    $ python dev.py test -s tree -- --tb=line  # `--` passes next args to pytest
    ```
    """  # noqa: E501

    ctx = CONTEXT

    verbose = Option(["--verbose", "-v"], default=False, is_flag=True, help="more verbosity")
    # removed doctests as currently not supported by _lib/_testutils.py
    # doctests = Option(['--doctests'], default=False)
    coverage = Option(
        ["--coverage", "-c"],
        default=False,
        is_flag=True,
        help=("report coverage of project code. " "HTML output goes under build/coverage"),
    )
    durations = Option(
        ["--durations", "-d"],
        default=None,
        metavar="NUM_TESTS",
        help="Show timing for the given number of slowest tests",
    )
    submodule = Option(
        ["--submodule", "-s"],
        default=None,
        metavar="MODULE_NAME",
        help="Submodule whose tests to run (cluster, constants, ...)",
    )
    tests = Option(
        ["--tests", "-t"], default=None, multiple=True, metavar="TESTS", help="Specify tests to run"
    )
    mode = Option(
        ["--mode", "-m"],
        default="fast",
        metavar="MODE",
        show_default=True,
        help=(
            "'fast', 'full', or something that could be passed to "
            "`pytest -m` as a marker expression"
        ),
    )
    parallel = Option(
        ["--parallel", "-j"],
        default=1,
        metavar="N_JOBS",
        help="Number of parallel jobs for testing",
    )
    # Argument can't have `help=`; used to consume all of `-- arg1 arg2 arg3`
    pytest_args = Argument(["pytest_args"], nargs=-1, metavar="PYTEST-ARGS", required=False)

    TASK_META = {
        "task_dep": ["build"],
    }

    @classmethod
    def sktree_tests(cls, args, pytest_args):
        dirs = Dirs(args)
        dirs.add_sys_path()
        print(f"sktree from development installed path at: {dirs.site}")

        # FIXME: support pos-args with doit
        extra_argv = pytest_args[:] if pytest_args else []
        if extra_argv and extra_argv[0] == "--":
            extra_argv = extra_argv[1:]

        if args.coverage:
            dst_dir = dirs.root / args.build_dir / "coverage"
            fn = dst_dir / "coverage_html.js"
            if dst_dir.is_dir() and fn.is_file():
                shutil.rmtree(dst_dir)
            extra_argv += ["--cov-report=html:" + str(dst_dir)]
            shutil.copyfile(dirs.root / ".coveragerc", dirs.site / ".coveragerc")

        if args.durations:
            extra_argv += ["--durations", args.durations]

        # convert options to test selection
        if args.submodule:
            tests = [PROJECT_MODULE + "." + args.submodule]
        elif args.tests:
            tests = args.tests
        else:
            tests = None

        runner, version, mod_path = get_test_runner(PROJECT_MODULE)
        # FIXME: changing CWD is not a good practice
        with working_dir(dirs.site):
            print(
                "Running tests for {} version:{}, installed at:{}".format(
                    PROJECT_MODULE, version, mod_path
                )
            )
            # runner verbosity - convert bool to int
            verbose = int(args.verbose) + 1
            result = runner(  # sktree._lib._testutils:PytestTester
                args.mode,
                verbose=verbose,
                extra_argv=extra_argv,
                doctests=False,
                coverage=args.coverage,
                tests=tests,
                parallel=args.parallel,
            )
        return result

    @classmethod
    def run(cls, pytest_args, **kwargs):
        """run unit-tests"""
        kwargs.update(cls.ctx.get())
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        args = Args(**kwargs)
        return cls.sktree_tests(args, pytest_args)


@cli.cls_cmd("bench")
class Bench(Task):
    """:wrench: Run benchmarks.

    \b
    ```python
     Examples:

    $ python dev.py bench -t integrate.SolveBVP
    $ python dev.py bench -t linalg.Norm
    $ python dev.py bench --compare main
    ```
    """

    ctx = CONTEXT
    TASK_META = {
        "task_dep": ["build"],
    }
    submodule = Option(
        ["--submodule", "-s"],
        default=None,
        metavar="SUBMODULE",
        help="Submodule whose tests to run (cluster, constants, ...)",
    )
    tests = Option(
        ["--tests", "-t"], default=None, multiple=True, metavar="TESTS", help="Specify tests to run"
    )
    compare = Option(
        ["--compare", "-c"],
        default=None,
        metavar="COMPARE",
        multiple=True,
        help=(
            "Compare benchmark results of current HEAD to BEFORE. "
            "Use an additional --bench COMMIT to override HEAD with COMMIT. "
            "Note that you need to commit your changes first!"
        ),
    )


###################
# linters


def emit_cmdstr(cmd):
    """Print the command that's being run to stdout

    Note: cannot use this in the below tasks (yet), because as is these command
    strings are always echoed to the console, even if the command isn't run
    (but for example the `build` command is run).
    """
    console = Console(theme=console_theme)
    # The [cmd] square brackets controls the font styling, typically in italics
    # to differentiate it from other stdout content
    console.print(f"{EMOJI.cmd} [cmd] {cmd}")


def task_lint():
    # Lint just the diff since branching off of main using a
    # stricter configuration.
    # emit_cmdstr(os.path.join('tools', 'lint.py') + ' --diff-against main')
    return {
        "basename": "lint",
        "actions": [str(Dirs().root / "tools" / "lint.py") + " --diff-against=main"],
        "doc": "Lint only files modified since last commit (stricter rules)",
    }


def task_unicode_check():
    # emit_cmdstr(os.path.join('tools', 'unicode-check.py'))
    return {
        "basename": "unicode-check",
        "actions": [str(Dirs().root / "tools" / "unicode-check.py")],
        "doc": "Check for disallowed Unicode characters in the sktree Python "
        "and Cython source code.",
    }


def task_check_test_name():
    # emit_cmdstr(os.path.join('tools', 'check_test_name.py'))
    return {
        "basename": "check-testname",
        "actions": [str(Dirs().root / "tools" / "check_test_name.py")],
        "doc": "Check tests are correctly named so that pytest runs them.",
    }


@cli.cls_cmd("lint")
class Lint:
    """:dash: Run linter on modified files and check for
    disallowed Unicode characters and possibly-invalid test names."""

    def run():
        run_doit_task(
            {
                "lint": {},
                "unicode-check": {},
                "check-testname": {},
            }
        )


@cli.cls_cmd("mypy")
class Mypy(Task):
    """:wrench: Run mypy on the codebase."""

    ctx = CONTEXT

    TASK_META = {
        "task_dep": ["build"],
    }

    @classmethod
    def run(cls, **kwargs):
        kwargs.update(cls.ctx.get())
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        args = Args(**kwargs)
        dirs = Dirs(args)

        try:
            import mypy.api
        except ImportError as e:
            raise RuntimeError(
                "Mypy not found. Please install it by running "
                "pip install -r mypy_requirements.txt from the repo root"
            ) from e

        config = dirs.root / "mypy.ini"
        check_path = PROJECT_MODULE

        with working_dir(dirs.site):
            # By default mypy won't color the output since it isn't being
            # invoked from a tty.
            os.environ["MYPY_FORCE_COLOR"] = "1"
            # Change to the site directory to make sure mypy doesn't pick
            # up any type stubs in the source tree.
            emit_cmdstr(f"mypy.api.run --config-file {config} {check_path}")
            report, errors, status = mypy.api.run(
                [
                    "--config-file",
                    str(config),
                    check_path,
                ]
            )
        print(report, end="")
        print(errors, end="", file=sys.stderr)
        return status == 0


##########################################
# DOC


@cli.cls_cmd("doc")
class Doc(Task):
    """:wrench: Build documentation.

    TARGETS: Sphinx build targets [default: 'html']
    """

    ctx = CONTEXT

    args = Argument(["args"], nargs=-1, metavar="TARGETS", required=False)
    list_targets = Option(
        ["--list-targets", "-t"],
        default=False,
        is_flag=True,
        help="List doc targets",
    )
    parallel = Option(
        ["--parallel", "-j"], default=1, metavar="N_JOBS", help="Number of parallel jobs"
    )

    @classmethod
    def task_meta(cls, list_targets, parallel, args, **kwargs):
        if list_targets:  # list MAKE targets, remove default target
            task_dep = []
            targets = ""
        else:
            task_dep = ["build"]
            targets = " ".join(args) if args else "html"

        kwargs.update(cls.ctx.get())
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        build_args = Args(**kwargs)
        dirs = Dirs(build_args)

        make_params = [f'PYTHON="{sys.executable}"']
        if parallel:
            make_params.append(f'SPHINXOPTS="-j{parallel}"')

        return {
            "actions": [
                # move to doc/ so local sktree does not get imported
                (
                    f'cd doc; env PYTHONPATH="{dirs.site}" '
                    f'make {" ".join(make_params)} {targets}'
                ),
            ],
            "task_dep": task_dep,
            "io": {"capture": False},
        }


@cli.cls_cmd("refguide-check")
class RefguideCheck(Task):
    """:wrench: Run refguide check."""

    ctx = CONTEXT

    submodule = Option(
        ["--submodule", "-s"],
        default=None,
        metavar="SUBMODULE",
        help="Submodule whose tests to run (cluster, constants, ...)",
    )
    verbose = Option(["--verbose", "-v"], default=False, is_flag=True, help="verbosity")

    @classmethod
    def task_meta(cls, **kwargs):
        kwargs.update(cls.ctx.get())
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        args = Args(**kwargs)
        dirs = Dirs(args)

        cmd = [f"{sys.executable}", str(dirs.root / "tools" / "refguide_check.py"), "--doctests"]
        if args.verbose:
            cmd += ["-vvv"]
        if args.submodule:
            cmd += [args.submodule]
        cmd_str = " ".join(cmd)
        return {
            "actions": [f"env PYTHONPATH={dirs.site} {cmd_str}"],
            "task_dep": ["build"],
            "io": {"capture": False},
        }


##########################################
# ENVS


@cli.cls_cmd("python")
class Python:
    """:wrench: Start a Python shell with PYTHONPATH set."""

    ctx = CONTEXT
    pythonpath = Option(
        ["--pythonpath", "-p"],
        metavar="PYTHONPATH",
        default=None,
        help="Paths to prepend to PYTHONPATH",
    )
    extra_argv = Argument(["extra_argv"], nargs=-1, metavar="ARGS", required=False)

    @classmethod
    def _setup(cls, pythonpath, **kwargs):
        vals = Build.opt_defaults()
        vals.update(kwargs)
        Build.run(add_path=True, **vals)
        if pythonpath:
            for p in reversed(pythonpath.split(os.pathsep)):
                sys.path.insert(0, p)

    @classmethod
    def run(cls, pythonpath, extra_argv=None, **kwargs):
        cls._setup(pythonpath, **kwargs)
        if extra_argv:
            # Don't use subprocess, since we don't want to include the
            # current path in PYTHONPATH.
            sys.argv = extra_argv
            with open(extra_argv[0]) as f:
                script = f.read()
            sys.modules["__main__"] = new_module("__main__")
            ns = dict(__name__="__main__", __file__=extra_argv[0])
            exec(script, ns)
        else:
            import code

            code.interact()


@cli.cls_cmd("ipython")
class Ipython(Python):
    """:wrench: Start IPython shell with PYTHONPATH set."""

    ctx = CONTEXT
    pythonpath = Python.pythonpath

    @classmethod
    def run(cls, pythonpath, **kwargs):
        cls._setup(pythonpath, **kwargs)
        import IPython

        IPython.embed(user_ns={})


@cli.cls_cmd("shell")
class Shell(Python):
    """:wrench: Start Unix shell with PYTHONPATH set."""

    ctx = CONTEXT
    pythonpath = Python.pythonpath
    extra_argv = Python.extra_argv

    @classmethod
    def run(cls, pythonpath, extra_argv, **kwargs):
        cls._setup(pythonpath, **kwargs)
        shell = os.environ.get("SHELL", "sh")
        print("Spawning a Unix shell...")
        os.execv(shell, [shell] + list(extra_argv))
        sys.exit(1)


@cli.command()
@click.argument("version_args", nargs=2)
@click.pass_obj
def notes(ctx_obj, version_args):
    """:ledger: Release notes and log generation.

    \b
    ```python
     Example:

    $ python dev.py notes v1.7.0 v1.8.0
    ```
    """
    if version_args:
        sys.argv = version_args
        log_start = sys.argv[0]
        log_end = sys.argv[1]
    cmd = f"python tools/write_release_and_log.py {log_start} {log_end}"
    click.echo(cmd)
    try:
        subprocess.run([cmd], check=True, shell=True)
    except subprocess.CalledProcessError:
        print("Error caught: Incorrect log start or log end version")


@cli.command()
@click.argument("revision_args", nargs=2)
@click.pass_obj
def authors(ctx_obj, revision_args):
    """:ledger: Generate list of authors who contributed within revision
    interval.

    \b
    ```python
    Example:

    $ python dev.py authors v1.7.0 v1.8.0
    ```
    """
    if revision_args:
        sys.argv = revision_args
        start_revision = sys.argv[0]
        end_revision = sys.argv[1]
    cmd = f"python tools/authors.py {start_revision}..{end_revision}"
    click.echo(cmd)
    try:
        subprocess.run([cmd], check=True, shell=True)
    except subprocess.CalledProcessError:
        print("Error caught: Incorrect revision start or revision end")


if __name__ == "__main__":
    cli()