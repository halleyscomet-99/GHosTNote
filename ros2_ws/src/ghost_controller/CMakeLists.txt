cmake_minimum_required(VERSION 3.22)
project(ghost_controller)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ghost_hardware REQUIRED)
include(${ghost_hardware_DIR}/../cmake/GhostMujoco.cmake)

add_library(${PROJECT_NAME} SHARED src/ghost_impedance_controller.cpp)
target_link_libraries(${PROJECT_NAME} ghost_mujoco ament_index_cpp::ament_index_cpp
  controller_interface::controller_interface hardware_interface::hardware_interface pluginlib::pluginlib
  rclcpp::rclcpp rclcpp_lifecycle::rclcpp_lifecycle realtime_tools::realtime_tools
  ${sensor_msgs_TARGETS} ${std_msgs_TARGETS} ${std_srvs_TARGETS} ${trajectory_msgs_TARGETS})
get_filename_component(_mj_libdir "${_mj_lib}" DIRECTORY)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${_mj_libdir}")

pluginlib_export_plugin_description_file(controller_interface ghost_controller.xml)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
ament_export_libraries(${PROJECT_NAME})
ament_package()
