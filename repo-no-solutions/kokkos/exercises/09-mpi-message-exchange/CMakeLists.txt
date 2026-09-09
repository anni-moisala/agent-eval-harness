# SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

project(MessageExchange LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED CONFIG)

add_executable(exchange exchange.cpp)

target_link_libraries(exchange PRIVATE Kokkos::kokkos)
target_link_libraries(exchange PRIVATE MPI::MPI_CXX)
