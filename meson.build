project('cato', 'cpp', 'cuda',
  version : '0.1.0',
  license : 'MIT',
  meson_version : '>= 0.60.0',
  default_options : ['cpp_std=c++17', 'cuda_std=c++17']
)

# Encontrar Python
py_mod = import('python')
py3 = py_mod.find_installation('python3', required: true)

# Encontrar pybind11
pybind11_dep = dependency('pybind11', required: true)

# Encontrar CUDA
cuda_dep = dependency('cuda', required: true)

# Configurar arquitectura CUDA
cuda_arch = '80'
add_project_arguments('-arch=compute_' + cuda_arch, language: 'cuda')
add_project_arguments('-arch=sm_' + cuda_arch, language: 'cuda')

# Incluir directorios
incdirs = include_directories(
  'include/device',
  'include/host'
)

# Fuentes del proyecto
sources = [
  'src/host/bindings/bindings.cpp',
  'src/host/bindings/devices_bindings.cpp',
  'src/device/utils/devices.cu',
  'src/device/memOps/wrappers/caTensorInitValueWrapper.cu',
  'src/device/memOps/wrappers/caTensorInitZeroWrapper.cu'
]

# Crear el módulo Python
py3.extension_module('cato',
  sources,
  include_directories : incdirs,
  dependencies : [pybind11_dep, cuda_dep],
  install : true
)
