cmake_minimum_required(VERSION 3.20)
project(strix_stream VERSION 0.1.0 LANGUAGES C CXX)

option(SS_CUDA "Build the CUDA backend" OFF)
option(SS_HIP "Build the HIP backend (Strix Halo, unified memory)" OFF)
option(SS_TESTS "Build the tests" ON)
option(SS_MUTATION_NO_SRC_WAIT "Test-only mutation: drop the exclusive same-forward source wait (must fail the CUDA test)" OFF)

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

add_library(strix_stream STATIC
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
  src/capi.cpp)
target_include_directories(strix_stream PUBLIC include PRIVATE src)
target_link_libraries(strix_stream PUBLIC Threads::Threads)

if(SS_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 120)  # RTX 4090, RTX PRO 6000
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  target_sources(strix_stream PRIVATE src/backend_cuda.cpp)
  target_compile_definitions(strix_stream PRIVATE SS_WITH_CUDA=1)
  if(SS_MUTATION_NO_SRC_WAIT)
    target_compile_definitions(strix_stream PRIVATE SS_MUTATION_NO_SRC_WAIT=1)
  endif()
  target_link_libraries(strix_stream PUBLIC CUDA::cudart)
endif()

if(SS_HIP)
  find_package(hip REQUIRED)
  target_sources(strix_stream PRIVATE src/backend_hip.cpp)
  target_compile_definitions(strix_stream PRIVATE SS_WITH_HIP=1)
  target_link_libraries(strix_stream PUBLIC hip::host)
endif()

add_executable(ss-plan tools/ss_plan.cpp)
target_link_libraries(ss-plan PRIVATE strix_stream)

add_executable(ss-recipe tools/ss_recipe.cpp)
target_include_directories(ss-recipe PRIVATE src)
target_link_libraries(ss-recipe PRIVATE strix_stream)

if(SS_TESTS)
  enable_testing()
  add_executable(ss-test-core tests/test_core.cpp)
  target_include_directories(ss-test-core PRIVATE src)
  target_link_libraries(ss-test-core PRIVATE strix_stream)
  add_test(NAME core COMMAND ss-test-core)

  add_executable(ss-test-runtime tests/test_runtime.cpp)
  target_include_directories(ss-test-runtime PRIVATE src)
  target_link_libraries(ss-test-runtime PRIVATE strix_stream)
  add_test(NAME runtime COMMAND ss-test-runtime)

  add_executable(ss-test-recipe tests/test_recipe.cpp)
  target_include_directories(ss-test-recipe PRIVATE src)
  target_link_libraries(ss-test-recipe PRIVATE strix_stream)
  add_test(NAME recipe COMMAND ss-test-recipe WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(ss-test-exclusive tests/test_exclusive.cpp)
  target_include_directories(ss-test-exclusive PRIVATE src)
  target_link_libraries(ss-test-exclusive PRIVATE strix_stream)
  add_test(NAME exclusive COMMAND ss-test-exclusive)

  add_executable(ss-test-readahead tests/test_readahead.cpp)
  target_include_directories(ss-test-readahead PRIVATE src)
  target_link_libraries(ss-test-readahead PRIVATE strix_stream)
  add_test(NAME readahead COMMAND ss-test-readahead)

  add_executable(ss-test-capi tests/test_capi.c)
  target_link_libraries(ss-test-capi PRIVATE strix_stream)
  set_target_properties(ss-test-capi PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME capi COMMAND ss-test-capi)

  # Shipped example recipes: each must validate; each structural negative fixture must be refused. The six
  # citation-only fixtures (x-* keys, which the core never reads) are checked by recipes/examples/validate_v1.py.
  file(GLOB ss_example_recipes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       recipes/examples/*.toml recipes/examples/pending/*.toml)
  foreach(r ${ss_example_recipes})
    add_test(NAME "example:${r}" COMMAND ss-recipe validate ${r} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
  set(ss_citation_only bad-bad-quote bad-dangling-cite bad-empty-evidence-list bad-layers-uncited
      bad-uncited-idle-preload bad-uncited-threshold)
  file(GLOB ss_bad_recipes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} recipes/examples/tests/bad-*.toml)
  foreach(r ${ss_bad_recipes})
    get_filename_component(n ${r} NAME_WE)
    if(NOT n IN_LIST ss_citation_only)
      add_test(NAME "refuse:${n}" COMMAND ss-recipe validate ${r} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties("refuse:${n}" PROPERTIES WILL_FAIL TRUE)
    endif()
  endforeach()

  if(SS_CUDA)
    add_executable(ss-probe-order tests/probe_order.cu)
    add_executable(ss-test-cuda tests/test_cuda.cu)
    target_include_directories(ss-test-cuda PRIVATE src)
    target_link_libraries(ss-test-cuda PRIVATE strix_stream CUDA::cudart)
  endif()
endif()
