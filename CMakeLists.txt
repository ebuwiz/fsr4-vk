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
set(FSR4_GENERATED "${CMAKE_BINARY_DIR}/generated")
option(FSR4_BUILD_UPSCALER "Build the FSR4 upscaler provider and embed its assets" ON)
if(FSR4_BUILD_UPSCALER)
set(FSR4_ASSET_ROOT "${CMAKE_SOURCE_DIR}/assets" CACHE PATH "Verified assets directory")
if(NOT EXISTS "${FSR4_ASSET_ROOT}/manifest.json")
    message(FATAL_ERROR
        "Generated assets are missing. Run tools/setup.py --dll /path/to/dlls. See README.md. "
        "For a custom asset directory, set FSR4_ASSET_ROOT.")
endif()
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
option(FFG4_EMBED_ASSETS "Embed the locally generated Frame Generation pack on Linux" ON)
set(FFG4_ASSET_ROOT "${CMAKE_SOURCE_DIR}/assets/frame-generation/pack" CACHE PATH
    "Verified Frame Generation asset pack")
# --- AMD FidelityFX Frame Generation provider: an independent second family.
# Separate output target (amd_fidelityfx_framegeneration_vk.{so,dll}). It links
# ONLY the FFX generic C API headers and Vulkan: no upscaler assets, no upscaler
# runtime, no model logic. It therefore cannot become an accidental merged provider
# with the upscaler, and the two outputs coexist independently.
add_library(amd_fidelityfx_framegeneration_vk SHARED
    src/framegeneration/provider.cpp src/framegeneration/capabilities.cpp
    src/framegeneration/assets.cpp)
target_include_directories(amd_fidelityfx_framegeneration_vk PRIVATE
    src src/framegeneration third_party/fidelityfx-api/api/include
    third_party/fidelityfx-api/framegeneration/include)
target_link_libraries(amd_fidelityfx_framegeneration_vk PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
set_target_properties(amd_fidelityfx_framegeneration_vk PROPERTIES
    PREFIX "" POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(FFG4_EMBED_ASSETS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT EXISTS "${FFG4_ASSET_ROOT}/manifest.json")
        message(FATAL_ERROR
            "Frame Generation assets are missing. Run tools/setup.py --dll /path/to/dlls, "
            "or configure with -DFFG4_EMBED_ASSETS=OFF and set FFG4_VK_ASSET_ROOT at runtime.")
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/assets/embed-framegeneration.py"
            "${FFG4_ASSET_ROOT}" "${FSR4_GENERATED}"
        RESULT_VARIABLE ffg_asset_result)
    if(NOT ffg_asset_result EQUAL 0)
        message(FATAL_ERROR "Frame Generation asset embedding/index generation failed")
    endif()
    enable_language(ASM)
    target_compile_definitions(amd_fidelityfx_framegeneration_vk PRIVATE FFG4_EMBEDDED_ASSETS)
    target_sources(amd_fidelityfx_framegeneration_vk PRIVATE
        "${FSR4_GENERATED}/framegeneration-assets.S")
    set_source_files_properties("${FSR4_GENERATED}/framegeneration-assets.S" PROPERTIES
        OBJECT_DEPENDS
        "${FSR4_GENERATED}/framegeneration-assets.bin;${FFG4_ASSET_ROOT}/manifest.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${FFG4_ASSET_ROOT}/manifest.json"
        "${CMAKE_SOURCE_DIR}/tools/assets/embed-framegeneration.py")
endif()
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
        third_party/fidelityfx-api/api/include
        third_party/fidelityfx-api/framegeneration/include)
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

# GPU-free self-test for the Frame Generation order/camera/create/generate predicates
# in src/framegeneration/framegeneration-checks.hpp — the exact checks the provider
# runs under FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING. No device, no assets, no
# algorithm; it pins the mandatory call order + camera contract so checker drift fails
# without a game-style dispatch.
add_executable(framegeneration-check-test tools/framegeneration-check-test.cpp)
target_include_directories(framegeneration-check-test PRIVATE
    src/framegeneration
    third_party/fidelityfx-api/framegeneration/include
    third_party/fidelityfx-api/api/include)
set_target_properties(framegeneration-check-test PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

# GPU-free consistency gate for the Frame Generation pass-plan substrate (roadmap Phase 0):
# it re-parses the pack's own pipelines.tsv with the executor's layout grammar and proves the
# independent reconstruction agrees with tools/extract/frame-generation/plan.py's fg-plan.tsv
# (474 compute modules, 1208 de-duplicated binding groups, a valid order scaffold). Pure std;
# no Vulkan, no device. Skips (exit 77) when the FG pack/plan have not been generated.
add_executable(framegeneration-plan-test tools/framegeneration-plan-test.cpp)
set_target_properties(framegeneration-plan-test PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)


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
    # GPU-free unit test for the documented SR 4.1.2 debug-checker predicates
    # (src/provider/upscaler-checks.hpp). No assets, no Vulkan, no device: it pins
    # the warning decision logic that the provider routes via providerMessage, so
    # drift in the checker is caught without a game-style dispatch.
    add_executable(upscaler-debugcheck-test tools/upscaler-debugcheck-test.cpp)
    target_include_directories(upscaler-debugcheck-test PRIVATE
        src third_party/fidelityfx-api/upscalers/include)
    set_target_properties(upscaler-debugcheck-test PROPERTIES CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    # GPU-free self-test for the deterministic FSR query helper math (quality->ratio,
    # display->render resolution, jitter phase-count + R2 offset) in
    # src/provider/upscaler-queries.hpp — the same functions the provider answers via
    # ffxQuery. Locks that math against drift with no device and no shader.
    add_executable(upscaler-query-test tools/upscaler-query-test.cpp)
    target_include_directories(upscaler-query-test PRIVATE src)
    set_target_properties(upscaler-query-test PROPERTIES CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
endif()
# Unified post-build gate target so IDEs and CI can run every Linux-feasible validator
# with one command: `cmake --build build/linux --target check`. It depends on the
# artifacts below (built first) then defers to tools/build/check.sh — the SAME gate
# tools/build/linux.sh runs — so the two entry points can never drift apart. EXCLUDE_FROM_ALL
# keeps the default build fast (the gate is not re-run on every incremental compile); it
# is UNIX-only because elf-assets.py rejects non-ELF images (the Windows path validates
# with windows-binary.py instead).
if(FSR4_BUILD_UPSCALER AND UNIX)
    add_custom_target(check
        COMMAND bash "${CMAKE_SOURCE_DIR}/tools/build/check.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Post-build validation gate (SPIR-V, ELF embed, FG ABI, debug-check predicates)"
        VERBATIM)
    add_dependencies(check
        amd_fidelityfx_upscaler_vk
        cap-probe
        descriptor-smoke
        upscaler-debugcheck-test)
    set_target_properties(check PROPERTIES EXCLUDE_FROM_ALL YES)
endif()
# Windows-side sibling of the ELF embed proof. `elf-assets.py` (in the `check` gate)
# validates the Linux .so; `windows-binary.py` is its exact PE counterpart: it walks the
# built amd_fidelityfx_upscaler_vk.dll's embedded RCDATA resources, hash-checks every
# payload against FSR4_ASSET_ROOT's manifest via the build's generated
# embedding-selection.json (embedded_assets + storage_index), and asserts the exported
# API is exactly {ffxCreateContext, ffxDestroyContext, ffxConfigure, ffxQuery, ffxDispatch}.
# Guarded to WIN32 so it is only DEFINED where that binary and selection actually exist
# (native MSVC or a MinGW-w64 cross build) and never misfires on a Linux configure. 8K
# is INCLUDED by the build (embed.py gets --include-8k), so --omit-8k is intentionally
# NOT passed. Registered EXCLUDE_FROM_ALL; CI runs `cmake --build --target check-win`.
if(FSR4_BUILD_UPSCALER AND WIN32 AND FSR4_EMBEDDED_ASSETS)
    add_custom_target(check-win
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/validate/windows-binary.py"
            "$<TARGET_FILE:amd_fidelityfx_upscaler_vk>"
            "${FSR4_ASSET_ROOT}"
            --report "${FSR4_GENERATED}/windows-asset-report.json"
            --embedding-selection "${FSR4_GENERATED}/embedding-selection.json"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Post-build Windows validation gate (PE embed + FFX API exports)"
        VERBATIM)
    add_dependencies(check-win amd_fidelityfx_upscaler_vk)
    set_target_properties(check-win PROPERTIES EXCLUDE_FROM_ALL YES)
endif()
# CTest entry points so `ctest --test-dir build/linux --output-on-failure` (and IDE
# test runners) discover the invariant checks directly, not only via the bash gate.
# GPU-free tests are expected to pass here; GPU-dependent ones auto-skip on exit
# code 77 (the tools' "no suitable AMD device" signal), so `ctest` stays green on a
# GPU-less CI box instead of failing. All these executables live inside
# FSR4_BUILD_UPSCALER, so the block is guarded to match.
if(FSR4_BUILD_UPSCALER)
    enable_testing()
    add_test(NAME upscaler-debugcheck COMMAND upscaler-debugcheck-test)
    add_test(NAME cap-probe-synthetic-gates COMMAND cap-probe)
    set_tests_properties(cap-probe-synthetic-gates PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME descriptor-smoke-transports-byte-identical COMMAND descriptor-smoke)
    set_tests_properties(descriptor-smoke-transports-byte-identical PROPERTIES
        SKIP_RETURN_CODE 77)
    add_test(NAME frame-generation-validators
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/validate/frame-generation/tests.py")
    set_tests_properties(frame-generation-validators PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        SKIP_REGULAR_EXPRESSION "not a PE")
    add_test(NAME framegeneration-check-order-camera COMMAND framegeneration-check-test)
    add_test(NAME upscaler-query-math COMMAND upscaler-query-test)
    # GPU-free FG pass-plan consistency (Phase 0): the C++ re-parse of pipelines.tsv must agree
    # with plan.py's fg-plan.tsv. Registered whenever the FG pack exists; the tool self-skips
    # (exit 77) if the plan has not been generated, so ctest stays green without an FG pack/plan.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/frame-generation/pack/pipelines.tsv")
        add_test(NAME framegeneration-plan-consistency COMMAND framegeneration-plan-test)
        set_tests_properties(framegeneration-plan-consistency PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
    # Pin the FG "all 474 compute modules compile under BOTH descriptor transports"
    # census as a regression. Registered only when the (gitignored, generated) FG asset
    # pack is present, so a checkout that has not run tools/convert/frame-generation is
    # green rather than red; a device-less host still skips cleanly via exit code 77.
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets/frame-generation/pack/pipelines.tsv")
        add_test(NAME framegeneration-compile-parity-474
            COMMAND framegeneration-exec --compile-pack --descriptor-buffer-parity --expect-compiled=474)
        set_tests_properties(framegeneration-compile-parity-474 PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        # On-device pass-plan drift gate: the executor's driver-backed binding census (the
        # SAME groups it builds real descriptor-set layouts from, then serializes 1208 of via
        # vkGetDescriptorEXT) must equal plan.py's fg-plan.tsv. Needs a suitable AMD device
        # (auto-skips via 77); if the plan is absent the executor reports and skips (77) too.
        add_test(NAME framegeneration-plan-drift
            COMMAND framegeneration-exec --compile-pack --descriptor-buffer-parity --expect-plan)
        set_tests_properties(framegeneration-plan-drift PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
    # On-device Frame Generation host-integration return-code LADDER: ffxCreateContext ->
    # order-before-configure (PARAMETER) -> GET_VERSIONS (pinned id) -> GPU-memory-usage honest
    # refusal (RUNTIME_ERROR) -> Configure (OK) -> Prepare (RUNTIME_ERROR) -> Generate
    # (RUNTIME_ERROR) -> present-callback (NO_PROVIDER) -> destroy (OK). The probe now ASSERTS
    # every stage's documented code (it previously only printed them and always returned 0), so
    # provider routing/refusal drift becomes a real gate. Needs a device offering the FG execution
    # contract; auto-skips (exit 77) on a device-less host, so ctest stays green here and turns
    # the whole boundary into a FAIL/return-code check on a suitable AMD device.
    if(FSR4_CAPABILITY_PROBE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME framegeneration-host-integration-ladder COMMAND framegeneration-probe)
        set_tests_properties(framegeneration-host-integration-ladder PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
endif()
if(FSR4_BUILD_UPSCALER AND MINGW)
    target_link_options(amd_fidelityfx_upscaler_vk PRIVATE -static $<$<CONFIG:Release>:-s>)
endif()
if(FSR4_BUILD_UPSCALER AND MSVC)
    set_property(TARGET fsr4_assets fsr4_runtime amd_fidelityfx_upscaler_vk
        PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
