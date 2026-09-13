SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

CXX=g++
CXX_FLAGS=-g -Og -Wall -Wextra -Wpedantic -std=c++23 -march=native
LD_FLAGS=-lhdf5

NAME=test
SOURCE=${NAME}.cpp

test: ${SOURCE}
> @${CXX} ${CXX_FLAGS} ${LD_FLAGS} ${SOURCE} -o ${NAME}
> @./${NAME}
> @rm -f test test_file.hdf5

clean:
> rm -f test test_file.hdf5
.PHONY: clean
