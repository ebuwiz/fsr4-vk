# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

cmake_minimum_required(VERSION 3.22.1)
project(fsr4_vk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(FSR4_CLANG_FORMAT NAMES clang-format-22 clang-format)
set(format_tool_args)
if(FSR4_CLANG_FORMAT)
    list(APPEND format_tool_args --clang-format "${FSR4_CLANG_FORMAT}")
endif()
add_custom_target(format
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/format.py" ${format_tool_args}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Formatting owned C/C++ sources (clang-format 22)"
    VERBATIM)
add_custom_target(format-check
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/format.py" --check ${format_tool_args}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking C/C++ formatting without changing files"
    VERBATIM)
set(Vulkan_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Headers/include" CACHE PATH
    "Vulkan headers directory")
if(NOT EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.h")
    message(FATAL_ERROR "Vulkan-Headers is missing. Run git submodule update --init --recursive.")
endif()
if(WIN32)
    set(vulkan_exports "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Loader/loader/vulkan-1.def")
    if(NOT EXISTS "${vulkan_exports}")
        message(FATAL_ERROR "Vulkan-Loader is missing. Run git submodule update --init --recursive.")
    endif()
    set(vulkan_import_library "${CMAKE_BINARY_DIR}/vulkan/vulkan-1.lib")
    if(MSVC)
        set(vulkan_import_command "${CMAKE_AR}" /nologo "/def:${vulkan_exports}"
            /machine:x64 "/out:${vulkan_import_library}")
    elseif(CMAKE_DLLTOOL)
        set(vulkan_import_command "${CMAKE_DLLTOOL}" -m i386:x86-64
            -d "${vulkan_exports}" -l "${vulkan_import_library}")
    else()
        message(FATAL_ERROR "A Windows import-library tool is required (lib.exe or CMAKE_DLLTOOL).")
    endif()
    add_custom_command(OUTPUT "${vulkan_import_library}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/vulkan"
        COMMAND ${vulkan_import_command}
        DEPENDS "${vulkan_exports}"
        COMMENT "Generating the Vulkan loader import library"
        VERBATIM)
    add_custom_target(fsr4_vulkan_import DEPENDS "${vulkan_import_library}")
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION "${vulkan_import_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}")
    add_dependencies(Vulkan::Vulkan fsr4_vulkan_import)
else()
    find_package(Vulkan REQUIRED)
endif()
option(FSR4_COMPRESS_SHADERS "Compress embedded shader payloads independently with Zstandard" ON)
option(FSR4_EMBEDDED_ASSETS "Embed self-contained provider assets (PE or ELF)" ON)
option(FSR4_BUILD_UPSCALER "Build the FSR4 upscaler provider and embed its assets" ON)
if(FSR4_BUILD_UPSCALER)
set(FSR4_ASSET_ROOT "${CMAKE_SOURCE_DIR}/assets" CACHE PATH "Verified assets directory")
if(NOT EXISTS "${FSR4_ASSET_ROOT}/manifest.json")
    message(FATAL_ERROR
        "Generated assets are missing. Run tools/setup.py --dll /path/to/dlls. See README.md. "
        "For a custom asset directory, set FSR4_ASSET_ROOT.")
endif()
set(FSR4_GENERATED "${CMAKE_BINARY_DIR}/generated")
set(FSR4_GLSLANG_VALIDATOR "" CACHE FILEPATH "Override the bundled host GLSL compiler")
set(FSR4_SPIRV_VAL "" CACHE FILEPATH "Override the bundled host SPIR-V validator")
set(overlay_tool_args)
set(overlay_tool_dependencies)
if(FSR4_GLSLANG_VALIDATOR)
    list(APPEND overlay_tool_args --glslang "${FSR4_GLSLANG_VALIDATOR}")
    list(APPEND overlay_tool_dependencies "${FSR4_GLSLANG_VALIDATOR}")
endif()
if(FSR4_SPIRV_VAL)
    list(APPEND overlay_tool_args --spirv-val "${FSR4_SPIRV_VAL}")
    list(APPEND overlay_tool_dependencies "${FSR4_SPIRV_VAL}")
endif()
set(overlay_header "${FSR4_GENERATED}/diagnostics/overlay_spv.hpp")
if(NOT FSR4_GLSLANG_VALIDATOR)
    list(APPEND overlay_tool_dependencies "${CMAKE_SOURCE_DIR}/third_party/glslang/CMakeLists.txt")
endif()
if(NOT FSR4_SPIRV_VAL)
    list(APPEND overlay_tool_dependencies
        "${CMAKE_SOURCE_DIR}/third_party/SPIRV-Tools/CMakeLists.txt"
        "${CMAKE_SOURCE_DIR}/third_party/SPIRV-Headers/CMakeLists.txt")
endif()
add_custom_command(OUTPUT "${overlay_header}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/overlay/build-shader.py"
        --source "${CMAKE_SOURCE_DIR}/src/diagnostics/overlay.comp"
        --output "${overlay_header}"
        ${overlay_tool_args}
    DEPENDS "${CMAKE_SOURCE_DIR}/src/diagnostics/overlay.comp"
        "${CMAKE_SOURCE_DIR}/tools/overlay/build-shader.py"
        "${CMAKE_SOURCE_DIR}/tools/convert/toolchain.py"
        ${overlay_tool_dependencies}
    COMMENT "Compiling and validating overlay shaders"
    VERBATIM)
set(asset_generator embed.py)
set(asset_generator_args)
if(FSR4_EMBEDDED_ASSETS)
    list(APPEND asset_generator_args --prepare --include-8k)
    if(FSR4_COMPRESS_SHADERS)
        list(APPEND asset_generator_args --compress-shaders)
    endif()
    if(NOT WIN32)
        list(APPEND asset_generator_args --native-only)
    endif()
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tools/assets/${asset_generator}"
    "${FSR4_ASSET_ROOT}" "${FSR4_GENERATED}" ${asset_generator_args}
    RESULT_VARIABLE asset_result)
if(NOT asset_result EQUAL 0)
    message(FATAL_ERROR "Asset verification/index generation failed. Run extraction, conversion and patching before building.")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${FSR4_ASSET_ROOT}/manifest.json"
    "${CMAKE_SOURCE_DIR}/tools/assets/embed.py"
    "${CMAKE_SOURCE_DIR}/tools/assets/catalog.py"
    "${CMAKE_SOURCE_DIR}/tools/assets/storage.py")
add_library(fsr4_assets STATIC src/assets/assets.cpp third_party/zstd/zstddeclib.c)
# The decoder supports C++, so it needs no separate C compiler.
set_source_files_properties(third_party/zstd/zstddeclib.c PROPERTIES LANGUAGE CXX)
target_include_directories(fsr4_assets PRIVATE third_party/zstd)
target_compile_definitions(fsr4_assets PRIVATE ZSTDLIB_VISIBILITY= ZSTDERRORLIB_VISIBILITY=)
set_target_properties(fsr4_assets PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fsr4_assets PUBLIC src "${FSR4_GENERATED}")
add_library(fsr4_runtime STATIC
    "${overlay_header}"
    src/runtime/context.cpp src/runtime/frame.cpp src/runtime/resources.cpp src/runtime/rcas.cpp src/runtime/color-space.cpp
    src/runtime/gpu-capabilities.cpp
    src/models/fp8.cpp src/models/int8.cpp
    src/diagnostics/logging.cpp src/diagnostics/capture.cpp src/diagnostics/overlay.cpp
    src/diagnostics/gpu-profile.cpp)
set_target_properties(fsr4_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fsr4_runtime PUBLIC fsr4_assets Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_include_directories(fsr4_runtime PUBLIC src PRIVATE
    third_party/fidelityfx-api/upscalers/include)
add_library(amd_fidelityfx_upscaler_vk SHARED
    src/provider/provider.cpp src/provider/capture.cpp)
target_link_libraries(amd_fidelityfx_upscaler_vk PRIVATE fsr4_runtime ${CMAKE_DL_LIBS})
target_include_directories(amd_fidelityfx_upscaler_vk PRIVATE
    third_party/fidelityfx-api/upscalers/include)
set_target_properties(fsr4_assets fsr4_runtime amd_fidelityfx_upscaler_vk PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
set_target_properties(amd_fidelityfx_upscaler_vk PROPERTIES PREFIX "")
if(FSR4_EMBEDDED_ASSETS)
    target_compile_definitions(fsr4_assets PRIVATE FSR4_EMBEDDED_ASSETS)
    target_compile_definitions(amd_fidelityfx_upscaler_vk PRIVATE FSR4_EMBEDDED_ASSETS)
    if(WIN32)
        enable_language(RC)
        target_sources(amd_fidelityfx_upscaler_vk PRIVATE "${FSR4_GENERATED}/assets.rc")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        enable_language(ASM)
        target_compile_definitions(fsr4_assets PRIVATE FSR4_ELF_ASSETS)
        target_compile_definitions(fsr4_runtime PRIVATE FSR4_NATIVE_ASSETS_ONLY)
        target_sources(fsr4_assets PRIVATE "${FSR4_GENERATED}/embedded-assets.S")
        set_source_files_properties("${FSR4_GENERATED}/embedded-assets.S" PROPERTIES
            OBJECT_DEPENDS "${FSR4_GENERATED}/embedded-assets.bin")
    else()
        message(FATAL_ERROR "Embedded assets currently support Windows PE and Linux ELF")
    endif()
endif()
endif()
option(FSR4_CAPABILITY_PROBE "Build the standalone device capability probes (POSIX hosts)" ON)
# --- AMD FidelityFX Frame Generation provider: an independent second family.
# Separate output target (amd_fidelityfx_framegeneration_vk.{so,dll}). It links
# ONLY the FFX generic C API headers and Vulkan: no upscaler assets, no upscaler
# runtime, no model logic. It therefore cannot become an accidental merged provider
# with the upscaler, and the two outputs coexist independently.
add_library(amd_fidelityfx_framegeneration_vk SHARED
    src/framegeneration/provider.cpp src/framegeneration/capabilities.cpp)
target_include_directories(amd_fidelityfx_framegeneration_vk PRIVATE
    src src/framegeneration third_party/fidelityfx-api/api/include)
target_link_libraries(amd_fidelityfx_framegeneration_vk PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
set_target_properties(amd_fidelityfx_framegeneration_vk PROPERTIES
    PREFIX "" POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(MINGW)
    target_link_options(amd_fidelityfx_framegeneration_vk PRIVATE -static $<$<CONFIG:Release>:-s>)
endif()
if(MSVC)
    set_property(TARGET amd_fidelityfx_framegeneration_vk
        PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# The on-device Frame Generation capability probe: creates a caller-owned device
# with exactly the measured execution contract enabled, feeds it to the provider
# ABI, and reports the create result + named blockers. It links the provider ABI
# but no assets, so it runs on a machine that is not running a game.
if(FSR4_CAPABILITY_PROBE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(framegeneration-probe tools/framegeneration-probe.cpp)
    target_link_libraries(framegeneration-probe PRIVATE
        amd_fidelityfx_framegeneration_vk Vulkan::Vulkan ${CMAKE_DL_LIBS})
    target_include_directories(framegeneration-probe PRIVATE
        third_party/fidelityfx-api/api/include)
    set_target_properties(framegeneration-probe PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
endif()

if(FSR4_CAPABILITY_PROBE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The Frame Generation execution + pipeline-cache benchmark harness: drives the
    # provider's OWN namespaced cache (fg4) with a real compiled compute pipeline
    # and measures cold->warm creation + a benchmark loop + profiling, plus loads
    # every pack SPIR-V as a driver shader module. It shares ONLY the header-only
    # pipeline-cache container (src/framegeneration/pipeline-cache.hpp) with the
    # provider, so there is one cache-format source of truth and no upscaler coupling.
    add_executable(framegeneration-exec tools/framegeneration-exec.cpp)
    target_link_libraries(framegeneration-exec PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
    target_include_directories(framegeneration-exec PRIVATE
        src/framegeneration tools)
    set_target_properties(framegeneration-exec PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
endif()


if(FSR4_BUILD_UPSCALER AND FSR4_CAPABILITY_PROBE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The probe links the capability snapshot and nothing else: no assets, no provider,
    # no VkInstance of the runtime's own. That is what lets the model ladder and the
    # device-class spellings in tools/cap-probe.cpp be gated on a machine that is not
    # running a game, and it is why the same resolveModel both code paths call has to
    # be the one helper rather than two similar ones.
    add_executable(cap-probe tools/cap-probe.cpp src/runtime/gpu-capabilities.cpp)
    target_include_directories(cap-probe PRIVATE src "${FSR4_GENERATED}"
        third_party/fidelityfx-api/upscalers/include)
    target_link_libraries(cap-probe PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
    set_target_properties(cap-probe PROPERTIES CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    add_executable(descriptor-smoke tools/descriptor-smoke.cpp)
    target_link_libraries(descriptor-smoke PRIVATE fsr4_runtime)
    target_include_directories(descriptor-smoke PRIVATE src third_party/fidelityfx-api/upscalers/include)
    set_target_properties(descriptor-smoke PROPERTIES CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
endif()
if(FSR4_BUILD_UPSCALER AND MINGW)
    target_link_options(amd_fidelityfx_upscaler_vk PRIVATE -static $<$<CONFIG:Release>:-s>)
endif()
if(FSR4_BUILD_UPSCALER AND MSVC)
    set_property(TARGET fsr4_assets fsr4_runtime amd_fidelityfx_upscaler_vk
        PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
