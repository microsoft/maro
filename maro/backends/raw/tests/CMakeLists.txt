# We are using lest for our cpp unit test frame
# see: https://github.com/martinmoene/lest
#
# To run tests
# mkdir build & cd build
# cmake ..
# cmake --build .
# ctest -VV -C Debug

cmake_minimum_required (VERSION 3.8)

project( raw_backend_tests)


set(RAR_BACKEND_SOURCES
    ../common.h
    ../bitset.h
    ../bitset.cpp
    ../attribute.h
    ../attribute.cpp
    ../attributestore.h
    ../attributestore.cpp
    ../frame.h
    ../frame.cpp
    ../snapshotlist.h
    ../snapshotlist.cpp
    ../backend.h
    ../backend.cpp
)

set(TEST_SOURCES
    test_bitset
    test_others
)


if( MSVC )
    set( std11 "" )
    set( cpp_options -W3 -EHsc)
else()
    set( std11 "--std=c++11" )
    set( cpp_options -Wall -Wextra -Wno-missing-braces)
endif()


# add_library(marorawbackend SHARED ${RAR_BACKEND_SOURCES})


enable_testing()


foreach( name ${TEST_SOURCES} )
    add_executable( ${name} ${name}.cpp ${RAR_BACKEND_SOURCES})

    target_compile_options( ${name} PUBLIC ${std11} ${cpp_options})
    add_test( NAME ${name} COMMAND ${name})
    set_property ( TEST ${name} PROPERTY LABEL raw ut)
endforeach()


