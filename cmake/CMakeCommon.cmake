# -----------------------------------------------------------------------------
# Check for and link to AVX instruction sets
# -----------------------------------------------------------------------------
macro(check_and_link_AVX target useavx)
	message(STATUS "Set instruction sets for ${target}")
	
	# Use cmake hardware checks to see whether AVX should be activated
	include(CheckCXXCompilerFlag)

	# Hardware accelations: SSE and AVX
	set(AXV_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX,-DUSE_AVX>)
	set(AXV2_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-DUSE_AVX2>)
	set(SSE2_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:SSE2,-DUSE_SSE2>)

	check_cxx_compiler_flag(${AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
	check_cxx_compiler_flag(${AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)

	if(${useavx} AND ${COMPILER_OPT_AVX2_SUPPORTED})
		MESSAGE( STATUS "Use AXV2")
		target_compile_options(${target} PRIVATE ${AXV2_CompileOption})
	elseif(${useavx} AND ${COMPILER_OPT_AVX_SUPPORTED})
		MESSAGE( STATUS "Use AXV")
		target_compile_options(${target} PRIVATE ${AXV_CompileOption})
	else()
		MESSAGE( STATUS "Use SSE2")
		target_compile_options(${target} PRIVATE ${SSE2_CompileOption})
	endif()

endmacro()
