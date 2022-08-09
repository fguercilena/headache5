# HEADACHE5 -> HiErArchical DAta format 5, C++ HEader library

A simple, single-header-only wrapper for the HDF5 C library for C++17. It
exposes some of the most commonly used HDF5 API features in a structured,
object-oriented way.

### Installation:
The library is header only, so simply copy or simlink the *headache5.hpp* header
somewhere you like and *#include* it in your C++ source files.

### Requirements:
- A compiler supporting C++17
- The C++ standard library
- The HDF5 library (to include and link against)

### Testing and examples:
The *test.cpp* file can be compiled and run by simply typing `make`. Besides
testing the main features of the library, you can take a look at it to as an
example of its usage.
