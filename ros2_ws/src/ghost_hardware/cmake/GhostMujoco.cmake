# libmujoco from the GHOST .venv (pip mujoco ships the headers and the shared library), so the C++ plugins
# run the very same MuJoCo build as the Python sim. Override with -DGHOST_MUJOCO_DIR=<.../site-packages/mujoco>.
if(NOT GHOST_MUJOCO_DIR AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/ghost_mujoco_dir.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/ghost_mujoco_dir.cmake")      # written by ghost_hardware's build
endif()
if(NOT GHOST_MUJOCO_DIR)
  file(GLOB _mj_dirs "${CMAKE_CURRENT_LIST_DIR}/../../../../.venv/lib/python3*/site-packages/mujoco")
  list(GET _mj_dirs 0 GHOST_MUJOCO_DIR)
endif()
get_filename_component(GHOST_MUJOCO_DIR "${GHOST_MUJOCO_DIR}" ABSOLUTE)
file(GLOB _mj_lib "${GHOST_MUJOCO_DIR}/libmujoco.so.*")
if(NOT _mj_lib OR NOT EXISTS "${GHOST_MUJOCO_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo not found in '${GHOST_MUJOCO_DIR}' (set GHOST_MUJOCO_DIR)")
endif()
add_library(ghost_mujoco SHARED IMPORTED)
set_target_properties(ghost_mujoco PROPERTIES
  IMPORTED_LOCATION "${_mj_lib}"
  INTERFACE_INCLUDE_DIRECTORIES "${GHOST_MUJOCO_DIR}/include")
message(STATUS "GHOST MuJoCo: ${_mj_lib}")
