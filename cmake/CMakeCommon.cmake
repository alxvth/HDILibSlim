# -----------------------------------------------------------------------------
# Check for and link to AVX instruction sets
# -----------------------------------------------------------------------------
macro(check_and_link_AVX target)
	message(STATUS "Set instruction sets for ${target}")
	
	# Use cmake hardware checks to see whether AVX should be activated
	include(CheckCXXCompilerFlag)

	if(MSVC)
		check_cxx_compiler_flag("/arch:AVX" COMPILER_OPT_AVX_SUPPORTED)
		if(${COMPILER_OPT_AVX_SUPPORTED})
			target_compile_options(${target} PRIVATE /arch:AVX)
		endif()
		
		check_cxx_compiler_flag("/arch:AVX2" COMPILER_OPT_AVX2_SUPPORTED)
		if(${COMPILER_OPT_AVX2_SUPPORTED})
			target_compile_options(${target} PRIVATE /arch:AVX2)
		endif()

	else()
		check_cxx_compiler_flag("-DUSE_AVX" COMPILER_OPT_AVX_SUPPORTED)
		if(${COMPILER_OPT_AVX_SUPPORTED})
			target_compile_options(${target} PRIVATE -DUSE_AVX)
		endif()
		
		check_cxx_compiler_flag("-DUSE_AVX2" COMPILER_OPT_AVX2_SUPPORTED)
		if(${COMPILER_OPT_AVX2_SUPPORTED})
			target_compile_options(${target} PRIVATE -DUSE_AVX2)
		endif()
	endif()

	if(${COMPILER_OPT_AVX_SUPPORTED})
		MESSAGE( STATUS "Use AXV")
	endif()
	if(${COMPILER_OPT_AVX2_SUPPORTED})
		MESSAGE( STATUS "Use AXV2")
	endif()
endmacro()
