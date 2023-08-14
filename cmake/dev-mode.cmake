# This file was generated with cmake-init by friendlyanon at https://github.com/friendlyanon/cmake-init

include(cmake/folders.cmake)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()


add_folders(Project)
