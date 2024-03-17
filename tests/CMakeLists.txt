# Locate GTest
find_package(GTest REQUIRED)
include(GoogleTest) # Include the GoogleTest module to register the test
include_directories(${GTEST_INCLUDE_DIRS})

# Add your test sources
file(GLOB_RECURSE TEST_SOURCES *.cpp)
add_executable(runTests ${TEST_SOURCES})

# Link runTests with what we want to test and the GTest and pthread library
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread gtest_main CudaLearningLib)

gtest_discover_tests(runTests) # Discover tests and generate the test target
