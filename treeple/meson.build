# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

c_args = []
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
    c_args += ['-Wno-format-extra-args', '-Wno-format']
  endif
endif

openmp_dep = dependency('OpenMP', language: 'c', required: false)

if not openmp_dep.found()
    warning(
'''
                ***********
                * WARNING *
                ***********

It seems that treeple cannot be built with OpenMP.

- Make sure you have followed the installation instructions:

    https://scikit-learn.org/dev/developers/advanced_installation.html

- If your compiler supports OpenMP but you still see this
  message, please submit a bug report at:

    https://github.com/treeple/treeple/issues

- The build will continue with OpenMP-based parallelism
  disabled. Note however that some estimators will run in
  sequential mode instead of leveraging thread-based
  parallelism.

                    ***
''')
endif

# NumPy include directory - needed in all submodules
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''
import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

cc = meson.get_compiler('c')

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
# py.extension_module('_name',
#   'source_fname',
#   numpy_nodepr_api)
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'

# TODO XXX: ENABLE WHEN DEBUGGING
boundscheck = 'False'

scikit_learn_cython_args = [
  '-X language_level=3', '-X boundscheck=' + boundscheck, '-X wraparound=False',
  '-X initializedcheck=False', '-X nonecheck=False', '-X cdivision=True',
  '-X profile=False',
  # Needed for cython imports across subpackages, e.g. cluster pyx that
  # cimports metrics pxd
  '--include-dir', meson.global_build_root(),
]
cython_c_args += scikit_learn_cython_args

c_args += numpy_nodepr_api

python_sources = [
  '__init__.py',
  'neighbors.py',
  'conftest.py',
]

py.install_sources(
  python_sources,
  subdir: 'treeple'
)

subdir('_lib')
subdir('ensemble')
subdir('experimental')
subdir('stats')
subdir('tests')
subdir('tree')
subdir('datasets')
