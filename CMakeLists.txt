cmake_minimum_required (VERSION 3.15)

# Set C++11 language standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

set(PROJECT "HDILibSlim")
PROJECT(${PROJECT})

set(HDILibSlim_VERSION "undefined" CACHE STRING "HDILibSlim Library version")
# Disallow in-source builds. 
# Build in sub dir e.g. source/build* is still allowed!
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}" AND NOT $ENV{CI})
   message(FATAL_ERROR "In-source builds are not allowed!\n"
    "Make sure to remove CMakeCache.txt and CMakeFiles/ "
    "from the source directory!")
endif()

# If the CMAKE_INSTALL_PREFIX has not been set by the user, set it to the build folder
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Default CMAKE_INSTALL_PREFIX detected. Setting to build directory.")
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Default install path" FORCE )
else()
    message(STATUS "Custom CMAKE_INSTALL_PREFIX detected. Leaving it untouched.")
endif()

if (CMAKE_GENERATOR STREQUAL Xcode)
    add_definitions(-DGL_SILENCE_DEPRECATION)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version")
endif()

message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")

# CMake Options
# Set the directory where CMake include files are to be found
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# cmake utils
include(CMakeCommon)

find_package(OpenMP)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	include_directories(${OpenMP_CXX_INCLUDE_DIRS})
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
elseif(CMAKE_GENERATOR STREQUAL Xcode)
	add_definitions( -D__USE_GCD__)
else()
    message(WARNING, "OpenMP not found!")
endif()

# add projects
add_subdirectory (hdi/utils)						# PROJECT_UTILS
add_subdirectory (hdi/data)							# PROJECT_DATA
add_subdirectory (hdi/dimensionality_reduction)		# PROJECT_DR

# build order
add_dependencies(hdidata hdiutils)
add_dependencies(hdidimensionalityreduction hdiutils)
add_dependencies(hdidimensionalityreduction hdidata)

# NOTE CMake installs are defined in the sub-projects.
# ***Create a HdpsCore package to support the find_package command**

# Helper macro for packaging
include(CMakePackageConfigHelpers)

# Generate the version file for use with find_package
set(hdilibslim_package_version "${HDILibSlim_VERSION}")
configure_file(${CMAKE_SOURCE_DIR}/cmake/ConfigVersion.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfigVersion.cmake" @ONLY)
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfigVersion.cmake"
#  VERSION "${HDILibSlim_VERSION}"
#  # When the hdps core is stable compatibility can 
#  # change to AnyNewerVersion or SameMajorVersion
#  COMPATIBILITY ExactVersion
#)

set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(CURRENT_BUILD_DIR "${CMAKE_BINARY_DIR}")

# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/HDILibSlimConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfig.cmake"
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR CURRENT_BUILD_DIR
  INSTALL_DESTINATION lib/cmake/HDILibSlim
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# install the package files files
install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfigVersion.cmake"
        DESTINATION lib/cmake/HDILibSlim
        COMPONENT HDISlim_PACKAGE
)