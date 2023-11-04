# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

cython_c_args = []
if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    bitness = run_command(
      '_build_utils/gcc_build_bitness.py',
      check: true
    ).stdout().strip()
    if bitness == '64'
      add_project_arguments('-DMS_WIN64', language: ['c', 'cpp'])
    endif
    # Silence warnings emitted by PyOS_snprintf for (%zd), see
    # https://github.com/rgommers/scipy/issues/118.
    # Use as c_args for extensions containing Cython code
    cython_c_args += ['-Wno-format-extra-args', '-Wno-format']
  endif
endif

# NumPy include directory - needed in all submodules
incdir_numpy = run_command(py3,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()
# inc_np = include_directories(incdir_numpy)

cc = meson.get_compiler('c')

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
# py3.extension_module('_name',
#   'source_fname',
#   numpy_nodepr_api)
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
cython_c_args += numpy_nodepr_api

python_sources = [
  '__init__.py',
  'neighbors.py',
  'utils.py',
  'conftest.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree'
)

cython_cli = find_program('_build_utils/cythoner.py')

cython_gen = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@'],
  output : '@BASENAME@.c')

cython_gen_cpp = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '--cplus'],
  output : '@BASENAME@.cpp')

c_undefined_ok = ['-Wno-maybe-uninitialized']

# Suppress warning for deprecated Numpy API.
# (Suppress warning messages emitted by #warning directives).
# Replace with numpy_nodepr_api after Cython 3.0 is out
cython_c_args += ['-Wno-cpp']
cython_cpp_args = cython_c_args

subdir('_lib')
subdir('ensemble')
subdir('experimental')
subdir('stats')
subdir('tests')
subdir('tree')
