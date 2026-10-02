cmake_minimum_required(VERSION 3.20)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29")
    cmake_policy(SET CMP0167 NEW)
endif()
project(avatar VERSION 0.1.0)
include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/options.local.cmake")
    include(${CMAKE_SOURCE_DIR}/cmake/options.local.cmake)
endif()

# default to optimized build, Debug is ~100x slower in Eigen/Pinocchio and misses 1 kHz
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
  message(STATUS "CMAKE_BUILD_TYPE not set -- defaulting to RelWithDebInfo")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING
    "Building Debug: Eigen/Pinocchio run unoptimised with assertions on, which "
    "costs roughly 100x in the control loop and WILL miss the 1 kHz deadline. "
    "Fine for stepping through logic, useless for any timing or latency work.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DCOAL_DISABLE_HPP_FCL_WARNINGS)

# ── Platform setup ────────────────────────────────────────────────────────────
if(WIN32)
    message(STATUS "Platform: Windows")
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4305)
    add_compile_options(/FI "${CMAKE_SOURCE_DIR}/include/win_preamble.hpp")
elseif(APPLE)
    message(STATUS "Platform: macOS")
elseif(UNIX)
    message(STATUS "Platform: Linux")
endif()

# ── Conda/miniforge prefix ────────────────────────────────────────────────────
# prefer conda env packages (e.g. yaml-cpp 0.8 over system 0.7)
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
elseif(EXISTS "/home/ceti/miniforge3")
    list(PREPEND CMAKE_PREFIX_PATH "/home/ceti/miniforge3")
endif()
if(UNIX AND EXISTS "/usr/local/lib/cmake")
    list(PREPEND CMAKE_PREFIX_PATH "/usr/local")
endif()

# ── Core dependencies ─────────────────────────────────────────────────────────
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Poco REQUIRED COMPONENTS Net Foundation)
find_package(msgpack-cxx REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)

# ── MuJoCo ────────────────────────────────────────────────────────────────────
if(BUILD_WITH_MUJOCO)
    find_package(tinyxml2 REQUIRED)
    find_package(glfw3 REQUIRED)

    if(NOT DEFINED MUJOCO_ROOT)
        if(DEFINED ENV{MUJOCO_ROOT})
            set(MUJOCO_ROOT "$ENV{MUJOCO_ROOT}")
        elseif(WIN32)
            set(MUJOCO_ROOT "C:/dev/mujoco-3.14.0")
        else()
            set(MUJOCO_ROOT "$ENV{HOME}/Workspace/tools/mujoco")
        endif()
    endif()
    message(STATUS "MuJoCo root: ${MUJOCO_ROOT}")

    add_library(mujoco SHARED IMPORTED GLOBAL)

    if(WIN32)
        set_target_properties(mujoco PROPERTIES
            IMPORTED_IMPLIB               "${MUJOCO_ROOT}/lib/mujoco.lib"
            IMPORTED_LOCATION             "${MUJOCO_ROOT}/bin/mujoco.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
        )
    elseif(APPLE)
        # flat install or framework bundle
        file(GLOB _mj_lib LIST_DIRECTORIES false
            "${MUJOCO_ROOT}/lib/libmujoco*.dylib"
            "${MUJOCO_ROOT}/Versions/A/libmujoco*.dylib"
        )
        list(GET _mj_lib 0 _mj_lib_path)
        if(EXISTS "${MUJOCO_ROOT}/include")
            set(_mj_include "${MUJOCO_ROOT}/include")
        else()
            set(_mj_include "${MUJOCO_ROOT}")
        endif()
        set_target_properties(mujoco PROPERTIES
            IMPORTED_LOCATION             "${_mj_lib_path}"
            INTERFACE_INCLUDE_DIRECTORIES "${_mj_include}"
        )
    else()
        file(GLOB _mj_lib LIST_DIRECTORIES false "${MUJOCO_ROOT}/lib/libmujoco.so.*")
        list(GET _mj_lib 0 _mj_lib_path)
        set_target_properties(mujoco PROPERTIES
            IMPORTED_LOCATION             "${_mj_lib_path}"
            INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
        )
    endif()

endif()

if(BUILD_WITH_MUJOCO)
    set(COAL_DISABLE_HPP_FCL_WARNINGS ON CACHE BOOL "" FORCE)
    find_package(pinocchio REQUIRED)
endif()

if(BUILD_WITH_FRANKA)
    find_package(Franka REQUIRED)
endif()

if(BUILD_STREAMER)
    if(WIN32 AND DEFINED ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64})
        set(ENV{PKG_CONFIG_PATH}
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/lib/pkgconfig")
        message(STATUS "GStreamer root: $ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}")
    endif()
    find_package(PkgConfig REQUIRED)
    if(BUILD_WITH_REALSENSE)
        find_package(realsense2 REQUIRED)
    endif()

    pkg_check_modules(GST REQUIRED
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-video-1.0
    )
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/avatar.cpp
    src/arm_control.cpp
    src/head_control.cpp
    src/MotionGenerator.cpp
    src/posture_optimizer.cpp
    src/network/udp_transport.cpp
    src/network/udp_reliable.cpp
    src/intention/intention_buffer.cpp
    src/intention/intention_recognizer.cpp
    src/twin/telemetry_forward.cpp
    src/twin/reconciler.cpp
)

if(BUILD_WITH_MUJOCO)
    list(APPEND SOURCES
        src/sim_env/simulation.cpp
        src/sim_env/robot.cpp
        src/sim_env/gripper.cpp
        src/sim_env/scene_builder.cpp
        src/sim_env/model.cpp
        src/sim_env/wrench_truth.cpp
        src/sim_env/tum_head_driver.cpp
    )
endif()

if(BUILD_WITH_FRANKA)
    list(APPEND SOURCES src/sim_env/tum_head_driver.cpp)
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

# ── Include dirs ──────────────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    yaml-cpp::yaml-cpp
    Eigen3::Eigen
    Boost::filesystem
    Poco::Net
    Poco::Foundation
    msgpack-cxx
)

if(WIN32)
    # winmm for 1ms timer resolution
    target_link_libraries(${PROJECT_NAME} PRIVATE ws2_32 mswsock winmm)
endif()

if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH   "/Library/Frameworks"
        INSTALL_RPATH "/Library/Frameworks"
    )
endif()

if(BUILD_WITH_MUJOCO)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        mujoco
        tinyxml2::tinyxml2
        glfw
        pinocchio::pinocchio
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_MUJOCO)

    # deploy the same tinyxml2 DLL we link against
    if(WIN32)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:tinyxml2::tinyxml2>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Deploying tinyxml2.dll"
        )
    endif()
endif()

if(BUILD_WITH_FRANKA)
    target_link_libraries(${PROJECT_NAME} PRIVATE Franka::Franka)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_FRANKA)
endif()

# ── Pipeline (camera streaming + HDF5 logging) ────────────────────────────────
if(BUILD_STREAMER)
    find_package(PkgConfig REQUIRED)
    if(UNIX AND NOT APPLE)
        pkg_check_modules(HDF5_SYS REQUIRED hdf5-serial)
        set(HDF5_INCLUDE_DIRS ${HDF5_SYS_INCLUDE_DIRS})
        set(HDF5_C_LIBRARIES  ${HDF5_SYS_LIBRARIES})
        set(HDF5_LIBRARY_DIRS ${HDF5_SYS_LIBRARY_DIRS})
    else()
        find_package(HDF5 REQUIRED COMPONENTS C)
    endif()
    if(BUILD_WITH_REALSENSE)
        find_package(realsense2 REQUIRED)
    endif()
    pkg_check_modules(GST REQUIRED
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-video-1.0
    )

    add_executable(avatar_pipeline
        src/pipeline/streamer_main.cpp
        src/pipeline/video_streamer.cpp
        src/pipeline/video_logger.cpp
        src/pipeline/camera_source.cpp
        src/pipeline/camera_channel.cpp
        src/pipeline/episode_controller.cpp
        src/pipeline/stream_quality_controller.cpp
    )

    if(UNIX AND NOT APPLE)
        target_sources(avatar_pipeline PRIVATE src/pipeline/v4l2_source.cpp)
        target_compile_definitions(avatar_pipeline PRIVATE WITH_V4L2)
        pkg_check_modules(TURBOJPEG libturbojpeg)
        if(TURBOJPEG_FOUND)
            target_compile_definitions(avatar_pipeline PRIVATE WITH_TURBOJPEG)
            target_include_directories(avatar_pipeline PRIVATE ${TURBOJPEG_INCLUDE_DIRS})
            target_link_libraries(avatar_pipeline PRIVATE ${TURBOJPEG_LIBRARIES})
        endif()
    endif()

    target_include_directories(avatar_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${GST_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
    )

    target_link_libraries(avatar_pipeline PRIVATE
        yaml-cpp::yaml-cpp
        msgpack-cxx
        ${GST_LIBRARIES}
        ${HDF5_C_LIBRARIES}
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(avatar_pipeline PRIVATE curl)
    endif()
    if(BUILD_WITH_REALSENSE)
        target_compile_definitions(avatar_pipeline PRIVATE WITH_REALSENSE)
        target_link_libraries(avatar_pipeline PRIVATE realsense2::realsense2)
        target_sources(avatar_pipeline PRIVATE src/pipeline/realsense_source.cpp)
    endif()

    if(WIN32)
        target_link_libraries(avatar_pipeline PRIVATE ws2_32 mswsock)
    endif()

    target_link_directories(avatar_pipeline PRIVATE
        ${GST_LIBRARY_DIRS}
        ${HDF5_LIBRARY_DIRS}
    )

    target_compile_options(avatar_pipeline PRIVATE
        ${GST_CFLAGS_OTHER}
    )

    target_compile_definitions(avatar_pipeline PRIVATE BUILD_STREAMER)

    if(WIN32 AND DEFINED ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64})
        target_link_directories(avatar_pipeline PRIVATE
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/lib")
        target_include_directories(avatar_pipeline PRIVATE
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/include"
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/include/gstreamer-1.0"
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/include/glib-2.0"
            "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}/lib/glib-2.0/include")
    endif()
endif()
