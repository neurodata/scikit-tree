extensions = [
  '_vector_hash',
]

py3.extension_module(ext,
  cython_gen_cpp.process(ext + '.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'sktree/tree/cpp',
)
