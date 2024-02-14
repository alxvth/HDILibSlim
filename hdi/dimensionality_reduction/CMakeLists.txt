set(PROJECT_DR "hdidimensionalityreduction")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT_DR} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT_DR} PRIVATE ${PROJECT_SOURCE_DIR})

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
target_include_directories(${PROJECT_DR} PRIVATE ${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
target_include_directories(${PROJECT_DR} PRIVATE ${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})

#set_target_properties(${PROJECT_DR} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

target_compile_features(${PROJECT_DR} PRIVATE cxx_std_17)

if(ENABLE_PID)
    set_target_properties(${PROJECT_DR} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(USE_OPENMP AND OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_DR} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_DR} PRIVATE Eigen3::Eigen)

# Hardware accelations: SSE and AVX
check_and_set_AVX(${PROJECT_DR} ${USE_AVX})
set_optimization_level(${PROJECT_DR} ${OPTIMIZATION_LEVEL})

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_DR} PRIVATE Threads::Threads)
endif(UNIX)

########### INSTALL ##############
if(INSTALL_HDILibSlim)
	install(TARGETS ${PROJECT_DR}
		EXPORT ${PROJECT_DR}Targets
		LIBRARY DESTINATION lib/$<CONFIGURATION>
		ARCHIVE DESTINATION lib/$<CONFIGURATION>
	#    PUBLIC_HEADER DESTINATION include/hdi/dimensionality_reduction
		COMPONENT hdidimensionalityreduction
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

    install(DIRECTORY "${HNSWLIB_INCLUDE_DIR}/hnswlib/" DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hnswlib" COMPONENT HNSW_HEADERS)
    install(CODE "foreach(ANNOY_HEADER annoylib.h kissrandom.h mman.h)
            execute_process(
                COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                    \"${ANNOY_INCLUDE_DIR}/\${ANNOY_HEADER}\" 
                    \"${CMAKE_INSTALL_PREFIX}/include/\${ANNOY_HEADER}\"
            )
        endforeach()"
        COMPONENT ANNOY_HEADERS
    )

	install(EXPORT ${PROJECT_DR}Targets
			FILE ${PROJECT_DR}Targets.cmake
			NAMESPACE HDISlim::
			DESTINATION lib/cmake/HDILibSlim
			COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
	)

	if(MSVC)
		# install Debug info files ("*.pdb") on windows
		install(FILES "$<TARGET_FILE_DIR:${PROJECT_DR}>/${PROJECT_DR}.pdb" 
			DESTINATION lib/$<CONFIGURATION> 
			OPTIONAL
		)
	endif()
endif()
