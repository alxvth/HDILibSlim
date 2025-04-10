set(PROJECT_DATA "hdislimdata")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  *.h)
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

add_library(${PROJECT_DATA} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT_DATA} PRIVATE ${PROJECT_SOURCE_DIR})

target_include_directories(${PROJECT_DATA} PUBLIC "${eigen_SOURCE_DIR}")

set_target_properties(${PROJECT_DATA} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

target_compile_features(${PROJECT_DATA} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_DATA} PRIVATE OpenMP::OpenMP_CXX)

# Hardware accelations: SSE and AVX
check_and_link_AVX(${PROJECT_DATA} ${HDILibSlim_ENABLE_AVX})

target_link_libraries(${PROJECT_DATA} PRIVATE ${OPENGL_LIBRARIES})

########### INSTALL ##############
if(HDILibSlim_INSTALL)
	install(TARGETS ${PROJECT_DATA}
		EXPORT ${PROJECT_DATA}Targets
		LIBRARY DESTINATION lib/$<CONFIGURATION>>
		ARCHIVE DESTINATION lib/$<CONFIGURATION>
		PUBLIC_HEADER DESTINATION include/hdi/data
		COMPONENT hdislimdata
	)

	install(EXPORT ${PROJECT_DATA}Targets
			FILE ${PROJECT_DATA}Targets.cmake
			NAMESPACE HDISlim::
			DESTINATION lib/cmake/HDILibSlim
			COMPONENT HDISLIMDATA_TARGET
	)

	if(MSVC)
		# install Debug info files ("*.pdb") on windows
		install(FILES "$<TARGET_FILE_DIR:${PROJECT_DATA}>/${PROJECT_DATA}.pdb" 
			DESTINATION lib/$<CONFIGURATION> 
			OPTIONAL
		)
	endif()
endif()