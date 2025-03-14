# -----------------------------------------------------------------------------
# Check for and link to AVX instruction sets
# -----------------------------------------------------------------------------
macro(check_and_link_AVX target useavx)
    message(STATUS "Set instruction sets for ${target}, USE_AVX is ${useavx}")

    if(${useavx})
        # Use cmake hardware checks to see whether AVX should be activated
        include(CheckCXXCompilerFlag)

        if(MSVC)
            set(Check_AXV_CompileOption /arch:AVX)
            set(Check_AXV2_CompileOption /arch:AVX2)
            set(Set_AXV_CompileOption /arch:AVX)
            set(Set_AXV2_CompileOption /arch:AVX2)
        else()
            set(Check_AXV_CompileOption -mavx)
            set(Check_AXV2_CompileOption -mavx2)
            set(Set_AXV_CompileOption -mavx -mfma -ftree-vectorize)
            set(Set_AXV2_CompileOption -mavx2 -mfma -ftree-vectorize)
        endif()

        if(NOT DEFINED COMPILER_OPT_AVX_SUPPORTED OR NOT DEFINED COMPILER_OPT_AVX2_SUPPORTED)
            check_cxx_compiler_flag(${Check_AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
            check_cxx_compiler_flag(${Check_AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)
        endif()

        if(${COMPILER_OPT_AVX2_SUPPORTED} AND ${ARGC} EQUAL 2)
            message( STATUS "Use AXV2 for ${target}: ${Set_AXV2_CompileOption}")
            target_compile_options(${target} PRIVATE ${Set_AXV2_CompileOption})
        elseif(${COMPILER_OPT_AVX_SUPPORTED})
            message( STATUS "Use AXV for ${target}: ${Set_AXV_CompileOption}")
            target_compile_options(${target} PRIVATE ${Set_AXV_CompileOption})
        endif()
    endif()
endmacro()
