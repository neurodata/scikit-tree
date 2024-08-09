tree_extension_metadata = {
  '_sklearn_splitter':
    {'sources': ['_sklearn_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_oblique_splitter':
    {'sources': ['_oblique_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_oblique_tree':
    {'sources': ['_oblique_tree.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_utils':
    {'sources': ['_utils.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_marginal':
    {'sources': ['_marginal.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
}

foreach ext_name, ext_dict : tree_extension_metadata
  py.extension_module(
    ext_name,
    ext_dict.get('sources'),
    dependencies: [np_dep],
    override_options : ext_dict.get('override_options', []),
    c_args: c_args,
    cython_args: cython_c_args,
    subdir: 'treeple/tree',
    install: true,
  )
endforeach

python_sources = [
  '__init__.py',
  '_classes.py',
  '_multiview.py',
  '_neighbors.py',
  '_honest_tree.py',
  '_marginalize.py',
  '_kernel.py',
]

py.install_sources(
  python_sources,
  subdir: 'treeple/tree'   # Folder relative to site-packages to install to
)

# TODO: comment in if we include tests
subdir('tests')
subdir('unsupervised')
subdir('manifold')
