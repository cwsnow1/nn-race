#!/bin/bash

CC=/opt/rocm/bin/hipcc
STD="20"
PACKAGES="-pthread"
DEFINES="-DNDEBUG"
FLAGS="-g -Wall -Werror -pedantic"
SRC_FILES="./*.cpp ./matrix_gpu.hip"
EXE="main"
LIBS="-lraylib -lm"

$CC -std=c++$STD $PACKAGES $DEFINES $FLAGS $SRC_FILES -o $EXE $LIBS

EXE="matrix_test"
SRC_FILES="./matrix.cpp ./matrix_gpu.hip ./test/matrix_test.cpp"
$CC -std=c++$STD $PACKAGES $DEFINES $FLAGS $SRC_FILES -o $EXE $LIBS
