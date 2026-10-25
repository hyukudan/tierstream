cmake_minimum_required(VERSION 3.20)
project(tierstream VERSION 0.2.0 LANGUAGES C CXX)

# Options. The strix-stream names (SS_*) still work for one release: a given SS_X seeds TS_X's default.
foreach(opt CUDA HIP TESTS MUTATION_NO_SRC_WAIT)
  if(DEFINED SS_${opt} AND NOT DEFINED TS_${opt})
    message(STATUS "tierstream: SS_${opt}=${SS_${opt}} -> TS_${opt} (the SS_* option names are kept for one release)")
    set(TS_${opt} ${SS_${opt}} CACHE BOOL "")
  endif()
endforeach()
option(TS_CUDA "Build the CUDA backend" OFF)
option(TS_HIP "Build the HIP backend (Strix Halo, unified memory)" OFF)
option(TS_TESTS "Build the tests" ON)
option(TS_MUTATION_NO_SRC_WAIT "Test-only mutation: drop the exclusive same-forward source wait (must fail the CUDA test)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W4> $<$<COMPILE_LANGUAGE:C,CXX>:/WX>
                      $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc> $<$<COMPILE_LANGUAGE:C,CXX>:/wd4324>
                      $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall> $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:C,CXX>:-Werror> $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()

find_package(Threads REQUIRED)

add_library(tierstream STATIC
  src/json.cpp
  src/toml.cpp
  src/xxh64.cpp
  src/sha256.cpp
  src/desc.cpp
  src/plan.cpp
  src/recipe.cpp
  src/recipe_resolve.cpp
  src/recipe_run.cpp
  src/resident_set.cpp
  src/file_reader.cpp
  src/backend_cpu.cpp
  src/stream_ctx.cpp
  src/capi.cpp
  src/ts_api.cpp)
target_include_directories(tierstream PUBLIC include PRIVATE src)
target_link_libraries(tierstream PUBLIC Threads::Threads)
# One release of the rename: the old CMake target name, and the old archive file next to the new one (engine build
# scripts link strix_stream.lib / libstrix_stream.a by path).
add_library(strix_stream ALIAS tierstream)
set(TS_OLD_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}strix_stream${CMAKE_STATIC_LIBRARY_SUFFIX})
add_custom_command(OUTPUT ${TS_OLD_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tierstream> ${TS_OLD_ARCHIVE}
  DEPENDS tierstream
  COMMENT "tierstream: the strix_stream archive name (kept for one release)")
add_custom_target(strix_stream_archive ALL DEPENDS ${TS_OLD_ARCHIVE})

if(TS_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 120)  # RTX 4090, RTX PRO 6000
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  target_sources(tierstream PRIVATE src/backend_cuda.cpp)
  target_compile_definitions(tierstream PRIVATE SS_WITH_CUDA=1)
  if(TS_MUTATION_NO_SRC_WAIT)
    target_compile_definitions(tierstream PRIVATE SS_MUTATION_NO_SRC_WAIT=1)
  endif()
  target_link_libraries(tierstream PUBLIC CUDA::cudart)
endif()

if(TS_HIP)
  find_package(hip REQUIRED)
  target_sources(tierstream PRIVATE src/backend_hip.cpp)
  target_compile_definitions(tierstream PRIVATE SS_WITH_HIP=1)
  target_link_libraries(tierstream PUBLIC hip::host)
endif()

# Tools: the new names, and the old ones (same sources, each printing its own name) for one release.
foreach(name ts-plan ss-plan)
  add_executable(${name} tools/ts_plan.cpp)
  target_compile_definitions(${name} PRIVATE TS_TOOL_NAME="${name}")
  target_link_libraries(${name} PRIVATE tierstream)
endforeach()
foreach(name ts-recipe ss-recipe)
  add_executable(${name} tools/ts_recipe.cpp)
  target_compile_definitions(${name} PRIVATE TS_TOOL_NAME="${name}")
  target_include_directories(${name} PRIVATE src)
  target_link_libraries(${name} PRIVATE tierstream)
endforeach()

if(TS_TESTS)
  enable_testing()
  foreach(t core runtime recipe exclusive readahead snapshot)
    add_executable(ts-test-${t} tests/test_${t}.cpp)
    target_include_directories(ts-test-${t} PRIVATE src)
    target_link_libraries(ts-test-${t} PRIVATE tierstream)
  endforeach()
  add_test(NAME core COMMAND ts-test-core)
  add_test(NAME runtime COMMAND ts-test-runtime)
  add_test(NAME recipe COMMAND ts-test-recipe WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME exclusive COMMAND ts-test-exclusive)
  add_test(NAME readahead COMMAND ts-test-readahead)
  add_test(NAME snapshot COMMAND ts-test-snapshot)

  add_executable(ts-test-capi tests/test_capi.c)
  target_link_libraries(ts-test-capi PRIVATE tierstream)
  set_target_properties(ts-test-capi PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME capi COMMAND ts-test-capi)

  # The rename's compatibility gates (wave4/wstream/rename/design.md §2.1, §4).
  # (1) the new header: both inclusion orders, member names, token pasting, C++ type identity, every ts_ wrapper
  add_executable(ts-test-names-c tests/test_ts_names.c)
  target_link_libraries(ts-test-names-c PRIVATE tierstream)
  set_target_properties(ts-test-names-c PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME ts-names-c COMMAND ts-test-names-c)
  add_executable(ts-test-names-cpp tests/test_ts_names.cpp)
  target_include_directories(ts-test-names-cpp PRIVATE tests)
  target_link_libraries(ts-test-names-cpp PRIVATE tierstream)
  add_test(NAME ts-names-cpp COMMAND ts-test-names-cpp)
  # (2) an unchanged pre-rename C test (the old header only) linked by FILE PATH against the old archive name
  add_executable(ts-test-old-abi tests/test_capi.c)
  add_dependencies(ts-test-old-abi strix_stream_archive)
  target_include_directories(ts-test-old-abi PRIVATE include)
  set(ts_old_abi_deps Threads::Threads)  # what an engine linking the file links itself
  if(TS_CUDA)
    list(APPEND ts_old_abi_deps CUDA::cudart)
  endif()
  if(TS_HIP)
    list(APPEND ts_old_abi_deps hip::host)
  endif()
  target_link_libraries(ts-test-old-abi PRIVATE ${TS_OLD_ARCHIVE} ${ts_old_abi_deps})
  set_target_properties(ts-test-old-abi PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME ts-old-abi COMMAND ts-test-old-abi)
  # (3) ts.h is generated from ss.h: regenerate and compare
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME ts-header-generated
             COMMAND ${Python3_EXECUTABLE} tools/gen_ts_header.py --check
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ts-old-symbols
             COMMAND ${Python3_EXECUTABLE} tools/check_old_symbols.py $<TARGET_FILE:tierstream>
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  # (4) the old tool names keep their exact error prefix (hyperloom strips "ss-recipe: ")
  add_test(NAME ts-prefix-ss-recipe COMMAND ss-recipe validate no-such-file.toml)
  set_tests_properties(ts-prefix-ss-recipe PROPERTIES PASS_REGULAR_EXPRESSION "^ss-recipe: recipe: cannot open recipe no-such-file.toml")
  add_test(NAME ts-prefix-ss-recipe-usage COMMAND ss-recipe)
  set_tests_properties(ts-prefix-ss-recipe-usage PROPERTIES PASS_REGULAR_EXPRESSION "^ss-recipe: ")
  add_test(NAME ts-prefix-ss-plan COMMAND ss-plan)
  set_tests_properties(ts-prefix-ss-plan PROPERTIES PASS_REGULAR_EXPRESSION "^usage: ss-plan ")
  add_test(NAME ts-prefix-ts-recipe COMMAND ts-recipe validate no-such-file.toml)
  set_tests_properties(ts-prefix-ts-recipe PROPERTIES PASS_REGULAR_EXPRESSION "^ts-recipe: recipe: cannot open recipe no-such-file.toml")
  # (5) the ss:: bridge for translation units built against src/ (minimax/rtx-prefill-ss calls ss::xxh64)
  add_executable(ts-test-ss-bridge tests/test_ss_bridge.cpp)
  target_include_directories(ts-test-ss-bridge PRIVATE src)
  target_link_libraries(ts-test-ss-bridge PRIVATE tierstream)
  add_test(NAME ts-ss-bridge COMMAND ts-test-ss-bridge)

  # Shipped example recipes: each must validate; each structural negative fixture must be refused. The six
  # citation-only fixtures (x-* keys, which the core never reads) are checked by recipes/examples/validate_v1.py.
  file(GLOB ts_example_recipes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       recipes/examples/*.toml recipes/examples/pending/*.toml)
  foreach(r ${ts_example_recipes})
    add_test(NAME "example:${r}" COMMAND ts-recipe validate ${r} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
  set(ts_citation_only bad-bad-quote bad-dangling-cite bad-empty-evidence-list bad-layers-uncited
      bad-uncited-idle-preload bad-uncited-threshold)
  file(GLOB ts_bad_recipes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} recipes/examples/tests/bad-*.toml)
  foreach(r ${ts_bad_recipes})
    get_filename_component(n ${r} NAME_WE)
    if(NOT n IN_LIST ts_citation_only)
      add_test(NAME "refuse:${n}" COMMAND ts-recipe validate ${r} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties("refuse:${n}" PROPERTIES WILL_FAIL TRUE)
    endif()
  endforeach()

  if(TS_CUDA)
    foreach(name ts-probe-order ss-probe-order)
      add_executable(${name} tests/probe_order.cu)
    endforeach()
    add_executable(ts-test-cuda tests/test_cuda.cu)
    target_include_directories(ts-test-cuda PRIVATE src)
    target_link_libraries(ts-test-cuda PRIVATE tierstream CUDA::cudart)
  endif()
endif()
