# The C++ side of the Neural Process MPC study, configured by setup.sh. Compiles the upstream's
# src_cpp library sources directly (the list and flags of its CMakeLists.txt: Release, C++17) rather
# than add_subdirectory, because the upstream sets NPMPC_PROJECT_ROOT from CMAKE_SOURCE_DIR; here it
# is the clone. Adds the replay drivers and builds every laOPT target that can run IPOPT twice:
#   <name>              against CasADi's IPOPT (the casadi 3.7.0 wheel's libipopt.3.dylib, 3.14.11)
#   <name>_scaly_ipopt  against Scaly's vendored IPOPT 3.14.19 (plugins/scaly-ipopt)
# Each binary's runtime search path lists its IPOPT directory first, because the wheel directory
# also holds a libipopt.dylib that would otherwise satisfy @rpath/libipopt.dylib.
cmake_minimum_required(VERSION 3.21)
project(np_mpc_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(NPMPC_UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/neural_process_mpc" CACHE PATH
    "The neural_process_mpc clone")
set(CASADI_PY_DIR "" CACHE PATH "site-packages/casadi of the venv: CasADi C++ SDK and CasADi's IPOPT")
set(SCALY_IPOPT_LIB "" CACHE FILEPATH "Scaly's vendored libipopt (plugins/scaly-ipopt/src/scaly_ipopt/lib)")
set(SCALY_IPOPT_INCLUDE_DIR "" CACHE PATH "Its C++ headers (plugins/scaly-ipopt/third_party/ipopt_install/include/coin-or)")
foreach(var CASADI_PY_DIR SCALY_IPOPT_LIB SCALY_IPOPT_INCLUDE_DIR)
  if(NOT EXISTS "${${var}}")
    message(FATAL_ERROR "${var} = '${${var}}' does not exist; run setup.sh")
  endif()
endforeach()

find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT Eigen3_VERSION VERSION_EQUAL "3.4.0")
  message(FATAL_ERROR "Eigen 3.4.0 expected, found ${Eigen3_VERSION} in ${Eigen3_DIR}")
endif()
find_package(laopt REQUIRED)
find_package(piqp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(casadi REQUIRED PATHS "${CASADI_PY_DIR}/cmake" NO_DEFAULT_PATH)
message(STATUS "Eigen ${Eigen3_VERSION}: ${Eigen3_DIR}; CasADi ${casadi_VERSION}: ${casadi_DIR}")

# The two IPOPT builds.
if(APPLE)
  set(casadi_ipopt_lib "${CASADI_PY_DIR}/libipopt.3.dylib")
else()
  set(casadi_ipopt_lib "${CASADI_PY_DIR}/libipopt.so.3")
endif()
add_library(ipopt_casadi SHARED IMPORTED)
set_target_properties(ipopt_casadi PROPERTIES
    IMPORTED_LOCATION "${casadi_ipopt_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${CASADI_PY_DIR}/include/coin-or")
add_library(ipopt_scaly SHARED IMPORTED)
set_target_properties(ipopt_scaly PROPERTIES
    IMPORTED_LOCATION "${SCALY_IPOPT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${SCALY_IPOPT_INCLUDE_DIR}")
get_filename_component(scaly_ipopt_dir "${SCALY_IPOPT_LIB}" DIRECTORY)
set(ipopt_dir_casadi "${CASADI_PY_DIR}")
set(ipopt_dir_scaly "${scaly_ipopt_dir}")
set(ipopt_label_casadi "casadi-wheel")
set(ipopt_label_scaly "scaly-vendored")

set(src "${NPMPC_UPSTREAM}/src_cpp")
add_library(npmpc STATIC
    ${src}/npmpc/dynamics/integrators/furuta_np.cpp
    ${src}/npmpc/dynamics/integrators/midpoint.cpp
    ${src}/npmpc/dynamics/furuta.cpp
    ${src}/npmpc/nps/casadi_utils.cpp
    ${src}/npmpc/nps/decoder.cpp
    ${src}/npmpc/nps/NeuralProcess.cpp
    ${src}/npmpc/nps/weights_io.cpp
    ${src}/npmpc/nps/eigen_utils.cpp
    ${src}/npmpc/nps/FurutaNPEigen.cpp
    ${src}/npmpc/mpc/problem/MPCBase.cpp
    ${src}/npmpc/mpc/problem/furuta_cost.cpp
    ${src}/npmpc/mpc/problem/FurutaNPMPC.cpp
    ${src}/npmpc/mpc/problem/FurutaMPC.cpp
    ${src}/npmpc/mpc/solver.cpp
    ${src}/npmpc/mpc/controller.cpp
    ${src}/npmpc/mpc/mpc_config_io.cpp
    ${src}/npmpc/mpc/qube_client.cpp
)
target_include_directories(npmpc PUBLIC ${src})
target_link_libraries(npmpc PUBLIC casadi::casadi yaml-cpp::yaml-cpp Eigen3::Eigen)

# npmpc_program(NAME SOURCE [IPOPT casadi|scaly]): an upstream-style executable. With IPOPT, it is a
# laOPT program (IPOPT and SQP/PIQP enabled, as the upstream's laOPT targets) linked to that build.
function(npmpc_program name source)
  cmake_parse_arguments(arg "" "IPOPT" "" ${ARGN})
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE npmpc)
  target_compile_definitions(${name} PRIVATE NPMPC_PROJECT_ROOT="${NPMPC_UPSTREAM}")
  set(rpath "${CASADI_PY_DIR}")
  if(arg_IPOPT)
    target_link_libraries(${name} PRIVATE laopt::laopt ipopt_${arg_IPOPT} piqp::piqp_header_only)
    target_compile_definitions(${name} PRIVATE LAOPT_WITH_IPOPT LAOPT_WITH_PIQP
        NPMPC_IPOPT_BUILD="${ipopt_label_${arg_IPOPT}}")
    set(rpath "${ipopt_dir_${arg_IPOPT}};${CASADI_PY_DIR}")
  endif()
  set_target_properties(${name} PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "${rpath}")
endfunction()

# The upstream programs (its CMakeLists.txt's executables).
foreach(name eval_furuta_np eval_furuta_np_eigen eval_furuta_mpc eval_furuta_np_casadi eval_furuta_eq_casadi
             run_furuta_np_casadi_client run_furuta_eq_casadi_client)
  npmpc_program(${name} ${src}/tests/${name}.cpp)
endforeach()
foreach(method np eq)
  npmpc_program(run_furuta_${method}_laopt_client ${src}/tests/run_furuta_${method}_laopt_client.cpp IPOPT casadi)

  # The laOPT clients hard-code SQP/PIQP; the IPOPT variant is a copy with only that line changed.
  set(client "${src}/tests/run_furuta_${method}_laopt_client.cpp")
  file(READ "${client}" text)
  set(published "constexpr SolverType kSolver = SolverType::SQP_PIQP;")
  string(FIND "${text}" "${published}" at)
  if(at EQUAL -1)
    message(FATAL_ERROR "'${published}' not found in ${client}")
  endif()
  string(REPLACE "${published}" "constexpr SolverType kSolver = SolverType::IPOPT; // baseline/CMakeLists.txt: was SQP_PIQP" text "${text}")
  set(copy "${CMAKE_CURRENT_BINARY_DIR}/run_furuta_${method}_laopt_ipopt_client.cpp")
  set(previous "")
  if(EXISTS "${copy}")
    file(READ "${copy}" previous)
  endif()
  if(NOT previous STREQUAL text)  # rewrite only on change, so a reconfigure does not rebuild it
    file(WRITE "${copy}" "${text}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${client}")

  foreach(ipopt casadi scaly)
    set(suffix "")
    if(ipopt STREQUAL "scaly")
      set(suffix "_scaly_ipopt")
    endif()
    npmpc_program(eval_furuta_${method}_laopt${suffix} ${src}/tests/eval_furuta_${method}_laopt.cpp IPOPT ${ipopt})
    npmpc_program(run_furuta_${method}_laopt_ipopt_client${suffix}
        "${CMAKE_CURRENT_BINARY_DIR}/run_furuta_${method}_laopt_ipopt_client.cpp" IPOPT ${ipopt})
  endforeach()
endforeach()

# The replay drivers.
npmpc_program(replay_laopt replay_laopt.cpp IPOPT casadi)
npmpc_program(replay_laopt_scaly_ipopt replay_laopt.cpp IPOPT scaly)
npmpc_program(replay_casadi replay_casadi.cpp)
foreach(name replay_laopt replay_laopt_scaly_ipopt replay_casadi)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
