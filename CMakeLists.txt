cmake_minimum_required(VERSION 3.13)

project(FastBDT)
set(FastBDT_VERSION_MAJOR 5)
set(FastBDT_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -Wextra -g -msse2")

configure_file(
  "${PROJECT_SOURCE_DIR}/include/FastBDT.h.in"
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
)

include_directories("${PROJECT_SOURCE_DIR}/include/" "${PROJECT_BINARY_DIR}/include/")

set(FastBDT_SOURCES
  "${PROJECT_SOURCE_DIR}/src/FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/Classifier.cxx"
  "${PROJECT_SOURCE_DIR}/src/FastBDT_IO.cxx"
)

set(FastBDT_TESTS
  "${PROJECT_SOURCE_DIR}/src/test_all.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_Performance.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_Classifier.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_IO.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_C_API.cxx"
)

set(FastBDT_HEADERS
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
  "${PROJECT_SOURCE_DIR}/include/Classifier.h"
  "${PROJECT_SOURCE_DIR}/include/FastBDT_IO.h"
)

set(FastBDT_CINTERFACE
  "${PROJECT_SOURCE_DIR}/src/FastBDT_C_API.cxx"
  "${PROJECT_SOURCE_DIR}/include/FastBDT_C_API.h"
)

set(FastBDT_Python
  "${PROJECT_SOURCE_DIR}/PyFastBDT/__init__.py"
  "${PROJECT_SOURCE_DIR}/PyFastBDT/FastBDT.py"
  "${PROJECT_SOURCE_DIR}/PyFastBDT/utility.py"
)

add_library(FastBDT_static STATIC ${FastBDT_SOURCES} ${FastBDT_HEADERS})
add_library(FastBDT_CInterface SHARED ${FastBDT_CINTERFACE} ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_CInterface)
add_library(FastBDT_shared SHARED ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_shared)

install(TARGETS FastBDT_static FastBDT_shared FastBDT_CInterface
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES ${FastBDT_HEADERS} DESTINATION include)

find_package(GTest)
if(GTEST_FOUND)
  add_executable(fastbdt-unittests ${FastBDT_TESTS} ${FastBDT_HEADERS} ${FastBDT_CINTERFACE})
  target_link_libraries(fastbdt-unittests ${GTEST_BOTH_LIBRARIES} FastBDT_static pthread)
  message(STATUS  ${GTEST_INCLUDE_DIRS})
  target_include_directories(fastbdt-unittests PUBLIC ${GTEST_INCLUDE_DIRS})

  install(TARGETS unittests DESTINATION bin)
else()
  message(STATUS "Could not find gtest installation, skip building unittests.")
endif()

find_program(PYTHON "python3")
if(PYTHON)
  # Configure files
  configure_file(
    "${PROJECT_SOURCE_DIR}/setup.py.in"
    "${PROJECT_BINARY_DIR}/setup.py"
  )
  configure_file(
    "${PROJECT_SOURCE_DIR}/pyproject.toml.in"
    "${PROJECT_BINARY_DIR}/pyproject.toml"
  )
  add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/PyFastBDT/.installed"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/PyFastBDT" "${PROJECT_BINARY_DIR}/PyFastBDT"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FastBDT_shared> "${PROJECT_BINARY_DIR}/PyFastBDT/"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FastBDT_CInterface> "${PROJECT_BINARY_DIR}/PyFastBDT/"
    COMMAND ${CMAKE_COMMAND} -E touch "${PROJECT_BINARY_DIR}/PyFastBDT/.installed"
    DEPENDS FastBDT_shared FastBDT_CInterface
  )
  add_custom_target(PyFastBDT ALL
    DEPENDS "${PROJECT_BINARY_DIR}/PyFastBDT/.installed"
  )
  add_custom_command(TARGET PyFastBDT POST_BUILD
    COMMAND ${PYTHON} -m pip install .
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
  )
else()
  message(STATUS "Could not find python3, skipping installation of python bindings.")
endif()

set(CPACK_PACKAGE_VERSION "${FastBDT_VERSION_MAJOR}.${FastBDT_VERSION_MINOR}")
set(CPACK_GENERATOR "RPM;DEB;TGZ")
set(CPACK_PACKAGE_NAME "FastBDT")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "software@belle2.org")
set(CPACK_PACKAGE_VENDOR "Private")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
SET(CPACK_DEBIAN_PACKAGE_SECTION "libs")
SET(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

include(CPack)
