cmake_minimum_required(VERSION 3.14)

project(lesstimateTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(lesstimate REQUIRED)
  enable_testing()
endif()

find_package(Catch2 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)

include(Catch)

# ---- Tests ----

add_executable(lesstimate_test source/lesstimate_test.cpp
               source/linear_regression.cpp)
                 
target_link_libraries(
    lesstimate_test PRIVATE
    LAPACK::LAPACK
    BLAS::BLAS
    lesstimate::lesstimate
    ${ARMADILLO_LIBRARIES}
    Catch2::Catch2WithMain
)
target_compile_features(lesstimate_test PRIVATE cxx_std_17)

catch_discover_tests(lesstimate_test)

# ---- End-of-file commands ----

add_folders(Test)
