set(PROJECT_DATA "hdidata")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  *.h)
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

add_library(${PROJECT_DATA} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT_DATA} PRIVATE ${PROJECT_SOURCE_DIR})

set_target_properties(${PROJECT_DATA} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

target_compile_features(${PROJECT_DATA} PRIVATE cxx_std_17)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_DATA} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Hardware accelations: SSE and AVX
check_and_link_AVX(${PROJECT_DATA} ${USE_AVX})

target_link_libraries(${PROJECT_DATA} PRIVATE ${OPENGL_LIBRARIES})

########### INSTALL ##############
if(INSTALL_HDILibSlim)
	install(TARGETS ${PROJECT_DATA}
		EXPORT ${PROJECT_DATA}Targets
		LIBRARY DESTINATION lib/$<CONFIGURATION>>
		ARCHIVE DESTINATION lib/$<CONFIGURATION>
		PUBLIC_HEADER DESTINATION include/hdi/data
		COMPONENT hdidata
	)

	# install Debug info files ("*.pdb") on windows
	install(FILES "$<TARGET_FILE_DIR:${PROJECT_DATA}>/${PROJECT_DATA}.pdb" 
		DESTINATION lib/$<CONFIGURATION> 
		OPTIONAL
	)

	install(EXPORT ${PROJECT_DATA}Targets
			FILE ${PROJECT_DATA}Targets.cmake
			NAMESPACE HDISlim::
			DESTINATION lib/cmake/HDILibSlim
			COMPONENT HDIDATA_TARGET
	)
endif()