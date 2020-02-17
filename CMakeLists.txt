cmake_minimum_required(VERSION 3.10)

project(pearl VERSION 0.0.1)

set(sourcefiles
src/cpp/pearl_bindings.cpp
src/cpp/pearl.cpp
src/cpp/lru_state.cpp
src/cpp/lossy_state_graph.cpp
)

set(include_dirs
third_party/streamDM/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(third_party/streamDM)
add_subdirectory(third_party/pybind11)
pybind11_add_module(pearl SHARED ${sourcefiles})

target_link_libraries(pearl PUBLIC streamDM)
target_include_directories(pearl PUBLIC ${include_dirs})
