cmake_minimum_required(VERSION 3.20)
project(gpt-oss-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ICU REQUIRED COMPONENTS uc i18n)

# Main binary (kept minimal; not linked with unfinished sources)
add_executable(gptoss src/main.cpp)
target_link_libraries(gptoss PRIVATE ICU::uc ICU::i18n)

# Tests
include(CTest)
if (BUILD_TESTING)
  add_executable(checkpoint_test tests/checkpoint_test.cpp)
  add_test(NAME checkpoint_test COMMAND checkpoint_test)
endif()
