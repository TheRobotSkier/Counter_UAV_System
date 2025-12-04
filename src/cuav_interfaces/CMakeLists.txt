cmake_minimum_required(VERSION 3.8)
project(cuav_interfaces)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(MSG_FILES
  "msg/DOA.msg"
)

set(SRV_FILES
  "srv/SpeedOfSound.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
