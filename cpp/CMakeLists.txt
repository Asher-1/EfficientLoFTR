cmake_minimum_required(VERSION 3.16)
project(eloftr_ggml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ELOFTR_GGML_VULKAN "Enable Vulkan backend" OFF)
option(ELOFTR_GGML_CUDA "Enable CUDA backend" OFF)

get_filename_component(ELOFTR_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_DIR "${ELOFTR_REPO_ROOT}/third_party/ggml" CACHE PATH "ggml source tree")
get_filename_component(GGML_DIR "${GGML_DIR}" ABSOLUTE)
if(EXISTS "${GGML_DIR}/CMakeLists.txt")
  set(GGML_VULKAN ${ELOFTR_GGML_VULKAN} CACHE BOOL "" FORCE)
  set(GGML_CUDA ${ELOFTR_GGML_CUDA} CACHE BOOL "" FORCE)
  add_subdirectory(${GGML_DIR} build-ggml EXCLUDE_FROM_ALL)
else()
  message(FATAL_ERROR
    "ggml not found at ${GGML_DIR}. Run: git submodule update --init third_party/ggml")
endif()

add_library(eloftr_ggml STATIC src/repvgg_ggml.cpp src/repvgg_backbone.cpp)
target_include_directories(eloftr_ggml PUBLIC include)
target_link_libraries(eloftr_ggml PUBLIC ggml)

add_executable(eloftr_backbone src/eloftr_backbone_main.cpp)
target_link_libraries(eloftr_backbone PRIVATE eloftr_ggml)
