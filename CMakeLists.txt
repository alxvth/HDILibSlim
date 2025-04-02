cmake_minimum_required (VERSION 3.24)

set(HDILibSlim_ENABLE_AVX ON CACHE BOOL "Use AVX instructions sets if available")
set(HDILibSlim_INSTALL ON CACHE BOOL "Whether to install the HDILibSlim")

set(PROJECT_HDILibSlim "HDILibSlim")
PROJECT(${PROJECT_HDILibSlim} C CXX)

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

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W2 /permissive- /Zc:__cplusplus")
endif()

# Dependencies
find_package(OpenMP)
find_package(Eigen3 CONFIG REQUIRED)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
endif()

# add projects
add_subdirectory (hdi/utils)				# PROJECT_UTILS
add_subdirectory (hdi/data)				# PROJECT_DATA
add_subdirectory (hdi/dimensionality_reduction)		# PROJECT_DR

if(HDILibSlim_INSTALL)
	# NOTE CMake installs are defined in the sub-projects.
	# ***Create a HdpsCore package to support the find_package command**

	# Helper macro for packaging
	include(CMakePackageConfigHelpers)

	# Generate the version file for use with find_package
	set(hdilibslim_package_version "${HDILibSlim_VERSION}")
	configure_file(${PROJECT_SOURCE_DIR}/cmake/ConfigVersion.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/HDILibSlimConfigVersion.cmake" @ONLY)
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
endif()
