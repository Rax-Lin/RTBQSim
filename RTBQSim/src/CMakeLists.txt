if(NOT TARGET nlohmann_json)
  # cmake-lint: disable=C0103
  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "")
  # cmake-lint: disable=C0103
  set(JSON_MultipleHeaders
      OFF
      CACHE INTERNAL "")
  # add nlohmann::json library
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/json" "extern/json" EXCLUDE_FROM_ALL)

  # the following sets the SYSTEM flag for the include dirs of the json libs to suppress warnings
  set_target_properties(
    nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                             $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
  )# cmake-lint: disable=C0307
endif()

if(NOT TARGET ${PROJECT_NAME})
  # main project library
  add_library(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Expression.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/CompoundOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Control.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Operation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/OpType.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/StandardOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/SymbolicOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/Parser.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/Scanner.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/Token.hpp
    ${PROJECT_SOURCE_DIR}/include/Permutation.hpp
    ${PROJECT_SOURCE_DIR}/include/QuantumComputation.hpp
    ${PROJECT_SOURCE_DIR}/include/QBatchSimulator.hpp
    ${PROJECT_SOURCE_DIR}/include/RTSpMSpMEngine.hpp
    # CircuitOptimizer.cpp
    operations/Expression.cpp
    operations/NonUnitaryOperation.cpp
    operations/Operation.cpp
    operations/StandardOperation.cpp
    operations/SymbolicOperation.cpp
    parsers/GRCSParser.cpp
    parsers/QASMParser.cpp
    parsers/QCParser.cpp
    parsers/RealParser.cpp
    parsers/TFCParser.cpp
    parsers/Parser.cpp
    parsers/Scanner.cpp
    QuantumComputation.cpp
    ${PROJECT_SOURCE_DIR}/src/rt_engine/RTSpMSpMEngine.cu)

  if(BQSIM_USE_RTSPMSPM)
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/rt_engine)

    if(CMAKE_CUDA_ARCHITECTURES)
      list(GET CMAKE_CUDA_ARCHITECTURES 0 RTSPMSPM_ARCH)
      set(RTSPMSPM_ARCH_FLAG "--gpu-architecture=sm_${RTSPMSPM_ARCH}")
    endif()
    set(RTSPMSPM_PTX ${CMAKE_CURRENT_BINARY_DIR}/rt_engine/optixSpMSpM.ptx)
    add_custom_command(
      OUTPUT ${RTSPMSPM_PTX}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rt_engine
      COMMAND ${CMAKE_CUDA_COMPILER} --ptx ${PROJECT_SOURCE_DIR}/src/rt_engine/optixSpMSpM.cu
              -o ${RTSPMSPM_PTX}
              -ccbin ${CMAKE_CUDA_HOST_COMPILER}
              ${RTSPMSPM_ARCH_FLAG}
              -DBQSIM_RT_FP64
              -I${PROJECT_SOURCE_DIR}/src/rt_engine
              -I${PROJECT_SOURCE_DIR}/include
              -I${CUDA_INCLUDE_DIRS}
              -I${OptiX_INSTALL_DIR}/include
              -I${OptiX_INSTALL_DIR}/SDK
      DEPENDS ${PROJECT_SOURCE_DIR}/src/rt_engine/optixSpMSpM.cu
              ${PROJECT_SOURCE_DIR}/src/rt_engine/optixSpMSpM.h
    )
    add_custom_target(rtspmspm_ptx DEPENDS ${RTSPMSPM_PTX})
    add_dependencies(${PROJECT_NAME} rtspmspm_ptx)
    target_compile_definitions(${PROJECT_NAME} PUBLIC BQSIM_RTSPMSPM_PTX=\"${RTSPMSPM_PTX}\")
  endif()

  # set include directories
  target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                    ${PROJECT_BINARY_DIR}/include
                                                    ${CUQUANTUM_ROOT}/include)

  target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)
  if(BQSIM_USE_RTSPMSPM)
    find_library(OPTIX_LIBRARY
      NAMES nvoptix libnvoptix.so.1
      PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/lib
        /lib
      PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
    )
    if(NOT OPTIX_LIBRARY)
      message(FATAL_ERROR "OPTIX_LIBRARY not found. Please install/mount libnvoptix.so or libnvoptix.so.1.")
    endif()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPTIX_LIBRARY})
    find_library(NVTOOLSEXT_LIBRARY
      NAMES nvToolsExt libnvToolsExt.so
      PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/lib
        /lib
      PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
    )
    if(NOT NVTOOLSEXT_LIBRARY)
      message(FATAL_ERROR "NVTOOLSEXT_LIBRARY not found. Please install/mount libnvToolsExt.")
    endif()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${NVTOOLSEXT_LIBRARY})
  endif()

  # # add options and warnings to the library
  # target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

  # add MQT alias
  add_library(MQT::Core ALIAS ${PROJECT_NAME})
  add_library(MQT::${OLD_PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()

set(JSON_MultipleHeaders
    OFF
    CACHE INTERNAL "")
set(TF_BUILD_TESTS
    OFF
    CACHE BOOL "")
set(TF_BUILD_EXAMPLES
    OFF
    CACHE BOOL "")
set(TF_BUILD_PROFILER
    OFF
    CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/taskflow" "extern/taskflow")
target_link_libraries(${PROJECT_NAME} PUBLIC Taskflow)
# the following sets the SYSTEM flag for the include dirs of the taskflow libs
# cmake-lint: disable=C0307
set_target_properties(
  Taskflow PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                      $<TARGET_PROPERTY:Taskflow,INTERFACE_INCLUDE_DIRECTORIES>)
