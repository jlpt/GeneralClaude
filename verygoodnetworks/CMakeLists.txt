cmake_minimum_required(VERSION 3.16)
project(Nworks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NWORKS_BUILD_WINDOW_APP "Build the raylib window app (downloads raylib if it isn't installed)" ON)

function(nworks_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow)
  endif()
endfunction()

# Simulation core: no graphics dependency.
add_library(nworks_core STATIC
  Nworks/random.cpp
  Nworks/neural_network.cpp
  Nworks/sensors.cpp
  Nworks/actuators.cpp
  Nworks/organism.cpp
  Nworks/world.cpp
  Nworks/headless.cpp
)
target_include_directories(nworks_core PUBLIC Nworks)
nworks_warnings(nworks_core)

# Window-free runner: nworks_headless --steps 1000000 --seed 1
add_executable(nworks_headless Nworks/headless_main.cpp)
target_link_libraries(nworks_headless PRIVATE nworks_core)
nworks_warnings(nworks_headless)

# Experiment tools (see experiments/README.md).
add_executable(nworks_exp experiments/exp.cpp)
target_link_libraries(nworks_exp PRIVATE nworks_core)
add_executable(nworks_assay experiments/assay.cpp)
target_link_libraries(nworks_assay PRIVATE nworks_core)

if(NWORKS_BUILD_WINDOW_APP)
  find_package(raylib 5.5 QUIET)
  if(NOT raylib_FOUND)
    include(FetchContent)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
      URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz)
    FetchContent_MakeAvailable(raylib)
  endif()

  add_executable(nworks Nworks/main.cpp Nworks/render.cpp)
  target_link_libraries(nworks PRIVATE nworks_core raylib)
  nworks_warnings(nworks)
endif()
