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
option(TS_RENAME_GATES "Also build and run the pre-rename (4dc53b0) C++ tests unchanged (needs git)" OFF)
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
# The copy is a step of the library's own build, so every build of tierstream (targeted or not, any configuration)
# refreshes it, in the directory of that configuration's archive.
# Computed as CMake places the archive (BYPRODUCTS cannot name the target's own $<TARGET_FILE_DIR>): the target's
# ARCHIVE_OUTPUT_DIRECTORY_<CONFIG> as given, else ARCHIVE_OUTPUT_DIRECTORY (+ /<CONFIG> under a multi-config
# generator unless it holds a generator expression), else the binary directory (+ /<CONFIG> likewise). Both
# properties are initialized from the CMAKE_* variables when the target is created. CTest ts-old-archive-location
# checks the result against $<TARGET_FILE_DIR:tierstream>.
# A relative directory is read against the binary directory, as CMake does. When the directory holds a generator
# expression its absoluteness is only known once evaluated, so it is decided by $<PATH:IS_ABSOLUTE> (CMake 3.24+).
function(ts_absolute_dir out d)
  string(GENEX_STRIP "${d}" plain)
  if(plain STREQUAL d)
    if(NOT IS_ABSOLUTE "${d}")
      set(d ${CMAKE_CURRENT_BINARY_DIR}/${d})
    endif()
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(d "$<IF:$<PATH:IS_ABSOLUTE,${d}>,${d},${CMAKE_CURRENT_BINARY_DIR}/${d}>")
  else()
    message(FATAL_ERROR "tierstream: an archive output directory with a generator expression needs CMake 3.24+ "
                        "(the strix_stream archive copy must know where the archive is)")
  endif()
  set(${out} "${d}" PARENT_SCOPE)
endfunction()
get_property(ts_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(ts_multi_config)
  set(ts_configs ${CMAKE_CONFIGURATION_TYPES})
else()
  set(ts_configs ${CMAKE_BUILD_TYPE})
endif()
get_target_property(ts_generic_dir tierstream ARCHIVE_OUTPUT_DIRECTORY)
set(ts_archive_dir "")
foreach(cfg ${ts_configs})
  string(TOUPPER ${cfg} ucfg)
  get_target_property(d tierstream ARCHIVE_OUTPUT_DIRECTORY_${ucfg})
  if(NOT d)
    if(ts_generic_dir)
      set(d ${ts_generic_dir})
    else()
      set(d ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    string(GENEX_STRIP "${d}" d_plain)
    if(ts_multi_config AND d_plain STREQUAL d)
      string(APPEND d "/${cfg}")
    endif()
  endif()
  ts_absolute_dir(d "${d}")
  if(ts_multi_config)
    string(APPEND ts_archive_dir "$<$<CONFIG:${cfg}>:${d}>")
  else()
    set(ts_archive_dir ${d})
  endif()
endforeach()
if(ts_archive_dir STREQUAL "")  # single-config with no CMAKE_BUILD_TYPE
  if(ts_generic_dir)
    set(ts_archive_dir ${ts_generic_dir})
  else()
    set(ts_archive_dir ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  ts_absolute_dir(ts_archive_dir "${ts_archive_dir}")
endif()
set(TS_OLD_ARCHIVE ${ts_archive_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}strix_stream${CMAKE_STATIC_LIBRARY_SUFFIX})
add_custom_command(TARGET tierstream POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tierstream> ${TS_OLD_ARCHIVE}
  BYPRODUCTS ${TS_OLD_ARCHIVE}
  COMMENT "tierstream: the strix_stream archive name (kept for one release)")

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
  add_dependencies(ts-test-old-abi tierstream)
  target_include_directories(ts-test-old-abi PRIVATE include)
  set(ts_old_abi_deps Threads::Threads)  # what an engine linking the file links itself
  if(TS_CUDA)
    list(APPEND ts_old_abi_deps CUDA::cudart)
  endif()
  if(TS_HIP)
    list(APPEND ts_old_abi_deps hip::host)
  endif()
  # by full path, in the directory the library really is in (never a bare name, which a linker reads as -l)
  target_link_libraries(ts-test-old-abi PRIVATE
    $<TARGET_FILE_DIR:tierstream>/${CMAKE_STATIC_LIBRARY_PREFIX}strix_stream${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${ts_old_abi_deps})
  set_target_properties(ts-test-old-abi PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME ts-old-abi COMMAND ts-test-old-abi)
  # the copy is where the library really is, and identical to it
  add_test(NAME ts-old-archive-location
           COMMAND ${CMAKE_COMMAND} -E compare_files $<TARGET_FILE:tierstream>
                   $<TARGET_FILE_DIR:tierstream>/${CMAKE_STATIC_LIBRARY_PREFIX}strix_stream${CMAKE_STATIC_LIBRARY_SUFFIX})
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
  # ss-plan prints what strix-stream's ss-plan printed (no prefix; the reference comparison is
  # wave4/wstream/rename/byte_gate.py); ts-plan prefixes every diagnostic.
  add_test(NAME ts-prefix-ss-plan COMMAND ss-plan)
  set_tests_properties(ts-prefix-ss-plan PROPERTIES PASS_REGULAR_EXPRESSION "^usage: ss-plan ")
  add_test(NAME ts-prefix-ss-plan-open COMMAND ss-plan no-such-file.json --device-gib 1)
  set_tests_properties(ts-prefix-ss-plan-open PROPERTIES PASS_REGULAR_EXPRESSION "^descriptor: ")
  add_test(NAME ts-prefix-ts-plan COMMAND ts-plan)
  set_tests_properties(ts-prefix-ts-plan PROPERTIES PASS_REGULAR_EXPRESSION "^ts-plan: usage: ts-plan ")
  add_test(NAME ts-prefix-ts-plan-arg COMMAND ts-plan d.json --bogus)
  set_tests_properties(ts-prefix-ts-plan-arg PROPERTIES PASS_REGULAR_EXPRESSION "^ts-plan: unknown argument --bogus")
  add_test(NAME ts-prefix-ts-plan-value COMMAND ts-plan d.json --ring)
  set_tests_properties(ts-prefix-ts-plan-value PROPERTIES PASS_REGULAR_EXPRESSION "^ts-plan: --ring needs a value")
  add_test(NAME ts-prefix-ts-plan-open COMMAND ts-plan no-such-file.json --device-gib 1)
  set_tests_properties(ts-prefix-ts-plan-open PROPERTIES PASS_REGULAR_EXPRESSION "^ts-plan: descriptor: ")
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

  # (6) opt-in: strix-stream's own C++ tests at the last pre-rename commit, built UNCHANGED (taken with git show)
  # against the renamed library, their ss:: names resolved through the xxh64.hpp bridge (force-included).
  if(TS_RENAME_GATES)
    find_package(Git REQUIRED)
    set(ts_base 4dc53b0)
    set(ts_old_dir ${CMAKE_CURRENT_BINARY_DIR}/rename-base-tests)
    file(MAKE_DIRECTORY ${ts_old_dir})
    foreach(f check.hpp fixture.hpp test_core.cpp test_runtime.cpp test_recipe.cpp test_exclusive.cpp
              test_readahead.cpp test_snapshot.cpp)
      execute_process(COMMAND ${GIT_EXECUTABLE} show ${ts_base}:tests/${f}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE ${ts_old_dir}/${f}
                      RESULT_VARIABLE ts_git_rc)
      if(NOT ts_git_rc EQUAL 0)
        message(FATAL_ERROR "TS_RENAME_GATES: git show ${ts_base}:tests/${f} failed")
      endif()
    endforeach()
    foreach(t core runtime recipe exclusive readahead snapshot)
      add_executable(ts-test-base-${t} ${ts_old_dir}/test_${t}.cpp)
      target_include_directories(ts-test-base-${t} PRIVATE src)
      target_compile_options(ts-test-base-${t} PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/FIxxh64.hpp,-includexxh64.hpp>)
      target_link_libraries(ts-test-base-${t} PRIVATE tierstream)
      add_test(NAME base-${t} COMMAND ts-test-base-${t} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
  endif()

  if(TS_CUDA)
    foreach(name ts-probe-order ss-probe-order)
      add_executable(${name} tests/probe_order.cu)
    endforeach()
    add_executable(ts-test-cuda tests/test_cuda.cu)
    target_include_directories(ts-test-cuda PRIVATE src)
    target_link_libraries(ts-test-cuda PRIVATE tierstream CUDA::cudart)
  endif()
endif()
