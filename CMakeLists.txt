cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    lesstimate
    VERSION 0.1.0
    DESCRIPTION "Optimize your models with lasso, scad, mcp, and other penalty functions"
    HOMEPAGE_URL "https://jhorzek.github.io/lesstimate/"
    LANGUAGES NONE
)

# specify dependencies:
# NOTE: The order of importing dependencies is important.
# Armadillo needs BLAS needs LAPACK
find_package(fmt REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(lesstimate_lesstimate INTERFACE)
add_library(lesstimate::lesstimate ALIAS lesstimate_lesstimate)

set_property(
    TARGET lesstimate_lesstimate PROPERTY
    EXPORT_NAME lesstimate
)

target_include_directories(
    lesstimate_lesstimate ${warning_guard}
    INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

# we don't want to use R -> set USE_R to 0
target_compile_definitions(lesstimate_lesstimate INTERFACE -DUSE_R=0)
target_compile_features(lesstimate_lesstimate INTERFACE cxx_std_17)

target_link_libraries(lesstimate_lesstimate INTERFACE 
                      fmt::fmt 
                      LAPACK::LAPACK
                      BLAS::BLAS
                      ${ARMADILLO_LIBRARIES})

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT lesstimate_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of lesstimate"
  )
endif()

include(cmake/dev-mode.cmake)
