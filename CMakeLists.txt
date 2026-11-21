cmake_minimum_required(VERSION 4.4)
project(Gurabo C)

set(CMAKE_C_STANDARD 11)

include_directories(include)

add_library(Gurabo STATIC src/gurabo.c
        include/error.h)
