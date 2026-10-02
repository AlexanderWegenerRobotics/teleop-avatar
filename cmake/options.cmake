option(BUILD_WITH_FRANKA  "Compile with libfranka for real robot" OFF)
option(BUILD_WITH_MUJOCO  "Compile with MuJoCo simulation" ON)
option(BUILD_WITH_TESTS   "Build test targets" ON)

if(WIN32)
    option(BUILD_STREAMER "Build video streamer" OFF)
else()
    option(BUILD_STREAMER "Build video streamer" ON)
endif()