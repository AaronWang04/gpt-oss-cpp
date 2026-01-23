cmake_minimum_required(VERSION 3.20)
project(gpt-oss-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main binary (kept minimal; not linked with unfinished sources)
add_executable(gptoss src/main.cpp)

# Tests
include(CTest)
if (BUILD_TESTING)
  add_executable(checkpoint_test tests/checkpoint_test.cpp)
  add_test(NAME checkpoint_test COMMAND checkpoint_test)
endif()
