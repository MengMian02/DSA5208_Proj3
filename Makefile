CXX ?= g++
MPICXX ?= mpicxx
BASE_FLAGS ?= -O3 -std=c++17 -DNDEBUG
WARN_FLAGS ?= -Wall -Wextra -Wpedantic

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null)
OPENMP_FLAGS ?= -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
OPENMP_LIBS ?= -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
OPENMP_FLAGS ?= -fopenmp
OPENMP_LIBS ?=
endif

.PHONY: all clean test

all: kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid

kmeans_serial: kmeans_serial.cpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@

kmeans_openmp: kmeans_openmp.cpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@

kmeans_mpi: kmeans_mpi.cpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@

kmeans_hybrid: kmeans_hybrid.cpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@

test: all
	python3 tests/run_tests.py

clean:
	rm -f kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid
