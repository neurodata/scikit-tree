"""Configure details for documentation with sphinx."""

import os
import re
import subprocess
import sys
from datetime import date

import sphinx_gallery  # noqa: F401
from sphinx_gallery.sorting import ExampleTitleSortKey

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(
    os.path.abspath(os.path.join(curdir, "../build-install/usr/lib/python3.9/site-packages/"))
)
sys.path.insert(0, os.path.abspath("sphinxext"))
import sktree
from sktree._lib.sklearn.ensemble._forest import ExtraTreesClassifier  # noqa
from sktree._lib.sklearn.ensemble._forest import ExtraTreesRegressor  # noqa
from sktree._lib.sklearn.ensemble._forest import RandomForestClassifier  # noqa
from sktree._lib.sklearn.ensemble._forest import RandomForestRegressor  # noqa

sys.path.append(os.path.abspath(os.path.join(curdir, "..", "sktree")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "sktree/_lib")))

# -- project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# General information about the project.
project = "scikit-tree"
author = "Adam Li <adam.li@columbia.edu>"
td = date.today()
copyright = f"2022-{td.year}, scikit-tree Developers. Last updated on {td.isoformat()}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = sktree.__version__
# The full version, including alpha/beta/rc tags.
release = version

gh_url = "https://github.com/neurodata/scikit-tree"

# -- general configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "6.0"

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_issues",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "doi_role",
    "add_toctree_functions",
    "allow_nan_estimators",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = False

# TODO: figure out why these are raising an error?
nitpick_ignore = [
    ("py:mod", "sktree.tree"),
]

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`. This can be set to 'py:obj' to
# make `filter` a cross-reference to the Python function “filter”.
default_role = "literal"

# -- options for HTML output -------------------------------------------------

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url=gh_url,
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}
# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html"],
}

html_context = {
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
    "versions_dropdown": {
        "dev": "v0.2 (devel)",
        "stable": "v0.1",
    },
}

# -- autosummary -------------------------------------------------------------
autosummary_generate = True

# -- autodoc -----------------------------------------------------------------
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = False
autodoc_default_options = {
    "inherited-members": None,
}

# -- numpydoc ----------------------------------------------------------------

# needed to prevent errors
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True

# x-ref
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "Path": "pathlib.Path",
    "bool": ":class:`python:bool`",
    "UnsupervisedDecisionTree": "sktree.tree.UnsupervisedDecisionTree",
    "ObliqueDecisionTreeClassifier": "sktree.tree.ObliqueDecisionTreeClassifier",
    "PatchObliqueDecisionTreeClassifier": "sktree.tree.PatchObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor": "sktree.tree.ObliqueDecisionTreeRegressor",
    "PatchObliqueDecisionTreeRegressor": "sktree.tree.PatchObliqueDecisionTreeRegressor",
    "UnsupervisedObliqueRandomForest": "sktree.ensemble.UnsupervisedObliqueRandomForest",
    "DecisionTreeClassifier": "sklearn.tree.DecisionTreeClassifier",
    "DecisionTreeRegressor": "sklearn.tree.DecisionTreeRegressor",
    "ExtraTreeRegressor": "sklearn.tree.ExtraTreeRegressor",
    "pipeline.Pipeline": "sklearn.pipeline.Pipeline",
    # "sklearn_fork.inspection.permutation_importance": "sklearn.inspection.permutation_importance",
}

numpydoc_xref_ignore = {
    "of",
    "or",
    "shape",
    "n_components",
    "n_pixels",
    "n_classes",
    "instance",
    "optional",
    "ArrayLike",
    "estimator",
    "pandas",
    "n_samples",
    "n_features",
    "n_features_new",
    "n_estimators",
    "n_outputs",
    "n_honest",
    "n_structure",
    "lists",
    "n_nodes",
    "X",
    "default",
    "sparse",
    "matrix",
    "Ignored",
    "UnsupervisedSplitter",
    # from sklearn
    "such",
    "arrays",
    "if",
    "dicts",
    "a",
    "Tree",
    "_type_",
    "MetadataRequest",
    "~utils.metadata_routing.MetadataRequest",
    "quantiles",
    "n_quantiles",
    "metric",
    "n_queries",
    "BaseForest",
    "BaseDecisionTree",
    "n_indexed",
    "n_queries",
    "n_features_x",
    "n_features_y",
    "n_features_z",
    "n_neighbors",
    "one",
    "joblib.parallel_backend",
}

# validation
# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT01",
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    "PR09",  # ending with a . is not necessary
}

numpydoc_validate = False
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # we currently don't document these properly (probably okay)
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
    r"\.__len__",
}

# -- sphinx-copybutton -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "sklearn": ("https://scikit-learn.org/dev", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
intersphinx_timeout = 5

# -- sphinx-gallery ----------------------------------------------------------
os.environ["_sktree_BUILDING_DOC"] = "true"
scrapers = ("matplotlib",)

compress_images = ("images", "thumbnails")
# let's make things easier on Windows users
# (on Linux and macOS it's easy enough to require this)
if sys.platform.startswith("win"):
    try:
        subprocess.check_call(["optipng", "--version"])
    except Exception:
        compress_images = ()

sphinx_gallery_conf = {
    "doc_module": ("sktree",),
    "reference_url": {
        "sktree": None,
    },
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    "image_scrapers": scrapers,
    "show_memory": not sys.platform.startswith(("win", "darwin")),
    "line_numbers": False,  # messes with style
    "within_subsection_order": ExampleTitleSortKey,
    "capture_repr": ("_repr_html_",),
    "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    "matplotlib_animations": True,
    "compress_images": compress_images,
    "filename_pattern": "^((?!sgskip).)*$",
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""

# -- Sphinx-issues -----------------------------------------------------------
issues_github_path = "neurodata/scikit-tree"

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

# def linkcode_resolve(domain: str, info: Dict[str, str]) -> Optional[str]:
#     """Determine the URL corresponding to a Python object.

#     Parameters
#     ----------
#     domain : str
#         One of 'py', 'c', 'cpp', 'javascript'.
#     info : dict
#         With keys "module" and "fullname".

#     Returns
#     -------
#     url : str | None
#         The code URL. If None, no link is added.
#     """
#     if domain != "py":
#         return None  # only document python objects

#     # retrieve pyobject and file
#     try:
#         module = import_module(info["module"])
#         pyobject = module
#         for elt in info["fullname"].split("."):
#             pyobject = getattr(pyobject, elt)
#         fname = inspect.getsourcefile(pyobject).replace("\\", "/")
#     except Exception:
#         # Either the object could not be loaded or the file was not found.
#         # For instance, properties will raise.
#         return None

#     # retrieve start/stop lines
#     source, start_line = inspect.getsourcelines(pyobject)
#     lines = "L%d-L%d" % (start_line, start_line + len(source) - 1)

#     # create URL
#     if "dev" in release:
#         branch = "main"
#     else:
#         return None  # alternatively, link to a maint/version branch
#     fname = fname.split("/scikit-tree/")[1]
#     url = f"{gh_url}/blob/{branch}/scikit-tree/{fname}#{lines}"
#     return url


def replace_sklearn_fork_with_sklearn(app, what, name, obj, options, lines):
    """
    This function replaces all instances of 'sklearn' with 'sklearn'
    in the docstring content.
    """
    # Convert the list of lines to a string
    content = "\n".join(lines)

    # Use regular expressions to replace 'sklearn_fork' with 'sklearn'
    content = re.sub(r"`pipeline.Pipeline", r"`~sklearn.pipeline.Pipeline", content)
    content = re.sub(r"`~utils.metadata_routing.MetadataRequest", r"``MetadataRequest``", content)
    content = re.sub(r"`np.quantile", r"`numpy.quantile", content)
    content = re.sub(r"`~np.quantile", r"`numpy.quantile", content)

    # Convert the modified string back to a list of lines
    lines[:] = content.split("\n")


def setup(app):
    app.connect("autodoc-process-docstring", replace_sklearn_fork_with_sklearn)
