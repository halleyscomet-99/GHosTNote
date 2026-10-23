cmake_minimum_required(VERSION 3.22)
project(ghost_hardware)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(OpenArmCAN REQUIRED)
find_package(std_srvs REQUIRED)
include(cmake/GhostMujoco.cmake)

add_library(${PROJECT_NAME} SHARED src/openarm_mit_system.cpp src/mujoco_mit_system.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME}
  OpenArmCAN::openarm_can ghost_mujoco
  hardware_interface::hardware_interface pluginlib::pluginlib rclcpp::rclcpp rclcpp_lifecycle::rclcpp_lifecycle
  ${std_srvs_TARGETS})
get_filename_component(_mj_libdir "${_mj_lib}" DIRECTORY)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${_mj_libdir}")

pluginlib_export_plugin_description_file(hardware_interface ghost_hardware.xml)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY cmake DESTINATION share/${PROJECT_NAME})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ghost_mujoco_dir.cmake" "set(GHOST_MUJOCO_DIR \"${GHOST_MUJOCO_DIR}\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ghost_mujoco_dir.cmake" DESTINATION share/${PROJECT_NAME}/cmake)
ament_export_libraries(${PROJECT_NAME})
ament_package()
