#! /bin/bash

clear
cmake -S. -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build
./test
echo ""
echo "Removing the 'test' executable, the 'build' directory and 'test_file.hdf5'..."
rm -r test build test_file.hdf5
echo "Done."
echo ""
