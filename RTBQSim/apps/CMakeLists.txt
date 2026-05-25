add_subdirectory("${PROJECT_SOURCE_DIR}/extern/cxxopts" "extern/cxxopts"
                 EXCLUDE_FROM_ALL)
# the following sets the SYSTEM flag for the include dirs of the cxxopts libs to
# cmake-lint: disable=C0307

set_target_properties(
  cxxopts PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                     $<TARGET_PROPERTY:cxxopts,INTERFACE_INCLUDE_DIRECTORIES>)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
link_libraries(Threads::Threads)

find_package(OpenMP REQUIRED)
add_compile_options(-fopenmp)

add_executable(RTBQSim ${CMAKE_CURRENT_SOURCE_DIR}/RTBQSim.cu)
target_link_libraries(RTBQSim PRIVATE ${PROJECT_NAME} cxxopts::cxxopts)
target_link_libraries(RTBQSim PRIVATE ${CUSTATEVEC_LIBRARY})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(RTBQSim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()
