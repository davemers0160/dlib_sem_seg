# ----------------------------------------------------------------------------
#  Root CMake file for DfD Graph Cuts
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j4
#


cmake_minimum_required(VERSION 2.8.12)

PROJECT(dlib_seg_train)
set(exe_name "seg_ex")



set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})

if(WIN32)
	include(D:/dlib/dlib/cmake)
else()
    if(${PLATFORM} STREQUAL "HPC")
        include(~/dlib/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "SL02319")
        include(/mnt/data/dlib-19.17/dlib/cmake)
    else()
        include(~/dlib-19.17/dlib/cmake)
    endif()
endif()

if(${PLATFORM} STREQUAL "Laptop")
    set(COMMON_ROOT "D:/Common")
    #set(DFD_ROOT "D:/IUPUI/DFD")
elseif(${PLATFORM} STREQUAL "MainGear")
    set(COMMON_ROOT "~/Common")
    #set(DFD_ROOT "~/DFD")
elseif(${PLATFORM} STREQUAL "HPC")
    set(COMMON_ROOT "$HOME/Common")
    #set(DFD_ROOT "$HOME/Projects/DFD")
elseif(${PLATFORM} STREQUAL "SL02319")
    set(COMMON_ROOT "/mnt/data/Common")
    #set(DFD_ROOT "/mnt/data/IUPUI/DfD")
else()
    set(COMMON_ROOT "D:/Common")
    #set(DFD_ROOT "D:/IUPUI/DFD")	
endif()

#include(${COMMON_ROOT}/cmake_utils/use_cpp_11.cmake)
#include(${COMMON_ROOT}/cmake_utils/use_sse_avx.cmake)

file(GLOB SOURCES "src/*.c*")

set(project_includes "include" "${COMMON_ROOT}/include")


message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

# find OpenCV
#find_package(OpenCV REQUIRED)
#if (OpenCV_FOUND)
#	set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
#else()
#	message("OpenCV not found, so we won't build the project.")
#endif()

message(" ")
message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")


include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib)

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()

endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")



