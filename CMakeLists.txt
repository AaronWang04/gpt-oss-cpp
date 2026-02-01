cmake_minimum_required(VERSION 3.20)
project(gpt-oss-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# i hate c++ build process
if (APPLE)
  foreach(prefix /opt/homebrew /usr/local)
    if (EXISTS "${prefix}/opt/icu4c")
      list(APPEND CMAKE_PREFIX_PATH "${prefix}/opt/icu4c")
      if (NOT DEFINED ICU_ROOT)
        set(ICU_ROOT "${prefix}/opt/icu4c")
      endif()
    endif()
  endforeach()
endif()

find_package(ICU REQUIRED COMPONENTS uc i18n)

# Main binary
add_executable(
  gptoss
  src/main.cpp
  src/checkpoint.cpp
  src/tokenizer.cpp
  src/model.cpp
  src/kernels.cpp
  src/utils.cpp
)
target_include_directories(gptoss PRIVATE includes)
target_link_libraries(gptoss PRIVATE ICU::uc ICU::i18n)

# Tests
include(CTest)
if (BUILD_TESTING)
  add_executable(checkpoint_test tests/checkpoint_test.cpp src/checkpoint.cpp src/utils.cpp)
  target_include_directories(checkpoint_test PRIVATE includes)
  add_test(NAME checkpoint_test COMMAND checkpoint_test)
endif()
