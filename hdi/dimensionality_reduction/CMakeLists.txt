set(PROJECT_DR "hdidimensionalityreduction")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT_DR} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT_DR} PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(${PROJECT_DR} PRIVATE glad::glad)

# check compiler intrinsics and link against them
check_and_link_AVX(${PROJECT_DR})

if(OpenMP_CXX_FOUND) 
	target_link_libraries(${PROJECT_DR} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_DR} PRIVATE Threads::Threads)
endif(UNIX)

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
target_include_directories(${PROJECT_DR} PRIVATE ${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
target_include_directories(${PROJECT_DR} PRIVATE ${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__USE_ANNOY__")


########### INSTALL ##############
#set_target_properties(${PROJECT_DR} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

install(TARGETS ${PROJECT_DR}
    EXPORT ${PROJECT_DR}Targets
    LIBRARY DESTINATION lib/$<CONFIGURATION>
    ARCHIVE DESTINATION lib/$<CONFIGURATION>
    COMPONENT hdidimensionalityreduction
)

# install Debug info files ("*.pdb") on windows
install(FILES "$<TARGET_FILE_DIR:${PROJECT_DR}>/${PROJECT_DR}.pdb" 
	DESTINATION lib/$<CONFIGURATION> 
	OPTIONAL
)

# Preserve the header hierarchy by explicit install
# the CMake PUBLIC_HEADER target property flattens it if used.
install(CODE "foreach(HEADER ${HeaderFiles})
        message(STATUS \"Installing: \${HEADER} to \${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction\")
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                \"${CMAKE_CURRENT_SOURCE_DIR}/\${HEADER}\" 
                \"${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction/\${HEADER}\"
        )

    endforeach()"
    COMPONENT PUBLIC_HEADERS
)

install(EXPORT ${PROJECT_DR}Targets
        FILE ${PROJECT_DR}Targets.cmake
        NAMESPACE HDISlim::
        DESTINATION lib/cmake/HDILibSlim
        COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
)
