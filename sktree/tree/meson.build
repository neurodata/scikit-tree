extensions = [
  '_sklearn_splitter',
  '_oblique_splitter',
  '_oblique_tree',
  '_utils',
  '_marginal',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    install: true,
    subdir: 'sktree/tree',
  )
endforeach

python_sources = [
  '__init__.py',
  '_classes.py',
  '_neighbors.py',
  '_honest_tree.py',
  '_marginalize.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree/tree'   # Folder relative to site-packages to install to
)

subdir('tests')
subdir('unsupervised')
subdir('manifold')
