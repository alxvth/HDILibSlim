set(PROJECT_UTILS "hdislimutils")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h glad/*.h glad/KHR/*.h)
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

add_library(${PROJECT_UTILS} STATIC ${HeaderFiles} ${SourceFiles} ${Resources} )

target_include_directories(${PROJECT_UTILS} PRIVATE ${PROJECT_SOURCE_DIR})

target_include_directories(${PROJECT_UTILS} PUBLIC "${eigen_SOURCE_DIR}")

#set_target_properties(${PROJECT_UTILS} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

target_compile_features(${PROJECT_UTILS} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_UTILS} PRIVATE OpenMP::OpenMP_CXX)

# Hardware accelations: SSE and AVX
check_and_link_AVX(${PROJECT_UTILS} ${HDILibSlim_ENABLE_AVX})

if (UNIX)
    target_link_libraries (${PROJECT_UTILS} PRIVATE ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
if(HDILibSlim_INSTALL)
	install(TARGETS ${PROJECT_UTILS}
		EXPORT ${PROJECT_UTILS}Targets
		LIBRARY DESTINATION lib/$<CONFIGURATION>
		ARCHIVE DESTINATION lib/$<CONFIGURATION>
	#    PUBLIC_HEADER DESTINATION include/hdi/utils
		COMPONENT hdislimutils
	)

	# Preserve the header hierarchy by explicit install
	# the CMake PUBLIC_HEADER target property flattens it if used.
	install(CODE "foreach(HEADER ${HeaderFiles})
			message(STATUS \"Installing: \${HEADER} to \${CMAKE_INSTALL_PREFIX}/include/hdi/utils\")
			execute_process(
				COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
					\"${CMAKE_CURRENT_SOURCE_DIR}/\${HEADER}\" 
					\"${CMAKE_INSTALL_PREFIX}/include/hdi/utils/\${HEADER}\"
			)

		endforeach()"
		COMPONENT PUBLIC_HEADERS
	)

	install(EXPORT ${PROJECT_UTILS}Targets
			FILE ${PROJECT_UTILS}Targets.cmake
			NAMESPACE HDISlim::
			DESTINATION lib/cmake/HDILibSlim
			COMPONENT HDISLIMUTILS_TARGET
	)

	if(MSVC)
		# install Debug info files ("*.pdb") on windows
		install(FILES "$<TARGET_FILE_DIR:${PROJECT_UTILS}>/${PROJECT_UTILS}.pdb" 
			DESTINATION lib/$<CONFIGURATION> 
			OPTIONAL
		)
	endif()

endif()
