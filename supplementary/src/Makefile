# Standalone build for the supplementary fixed-total/weak-scaling K-means.
# Run from supplementary/: make -f src/Makefile all
# Binaries are written to supplementary/ so scripts/*.py find them via
# bencher.ROOT. The main repository Makefile is not modified.
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
.PHONY: all clean
all: kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid
kmeans_serial: src/kmeans_serial.cpp src/kmeans_common.hpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@
kmeans_openmp: src/kmeans_openmp.cpp src/kmeans_common.hpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@
kmeans_mpi: src/kmeans_mpi.cpp src/kmeans_common.hpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@
kmeans_hybrid: src/kmeans_hybrid.cpp src/kmeans_common.hpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@
clean:
	rm -f kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid
