cmake_minimum_required(VERSION 3.18)
project(cato LANGUAGES CXX CUDA)

# Estándares
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Encuentra pybind11
find_package(pybind11 CONFIG REQUIRED)

# Incluye directorios
include_directories(
    ${CMAKE_SOURCE_DIR}/include/device
    ${CMAKE_SOURCE_DIR}/include/host
    ${pybind11_INCLUDE_DIRS}
)

# Fuentes
file(GLOB_RECURSE BINDINGS_SRC
    src/host/bindings/*.cpp
)
file(GLOB_RECURSE DEVICE_WRAPPERS_SRC
    src/device/memOps/wrappers/*.cu
    src/device/utils/*.cu
)

# Crea el módulo Python
pybind11_add_module(cato
    ${BINDINGS_SRC}
    ${DEVICE_WRAPPERS_SRC}
)

# Banderas de compilación específicas
if (MSVC)
    target_compile_options(cato PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
else()
    target_compile_options(cato PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endif()

# Propiedades
set_target_properties(cato PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Linkea CUDA
target_link_libraries(cato PRIVATE cudart)

# Debugging
message(STATUS "Building cato with CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "PYBIND11_INCLUDE_DIRS=${pybind11_INCLUDE_DIRS}")