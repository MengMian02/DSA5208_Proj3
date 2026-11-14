CXX ?= g++
MPICXX ?= mpicxx
PYTHON ?= python3
BASE_FLAGS ?= -O3 -std=c++17 -DNDEBUG
WARN_FLAGS ?= -Wall -Wextra -Wpedantic

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null)
OPENMP_FLAGS ?= -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
OPENMP_LIBS ?= -L$(BREW_PREFIX)/opt/libomp/lib -lomp
METAL_FLAGS ?= -fobjc-arc
METAL_LIBS ?= -framework Metal -framework Foundation
CPU_TARGETS := kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid
GPU_TARGETS := kmeans_metal
else
OPENMP_FLAGS ?= -fopenmp
OPENMP_LIBS ?=
CPU_TARGETS := kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid
GPU_TARGETS :=
endif

.PHONY: all cpu gpu clean test smoke

all: cpu gpu

cpu: $(CPU_TARGETS)

gpu: $(GPU_TARGETS)

kmeans_serial: kmeans_serial.cpp kmeans_common.hpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@

kmeans_openmp: kmeans_openmp.cpp kmeans_common.hpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@

kmeans_mpi: kmeans_mpi.cpp kmeans_common.hpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $< -o $@

kmeans_hybrid: kmeans_hybrid.cpp kmeans_common.hpp
	$(MPICXX) $(BASE_FLAGS) $(WARN_FLAGS) $(OPENMP_FLAGS) $< $(OPENMP_LIBS) -o $@

# The Metal shader is compiled from an embedded source string at run time, so
# the build needs only the Command Line Tools (no xcrun metal / full Xcode).
gpu/metal/kernels_source.h: gpu/metal/kmeans.metal gpu/metal/embed_source.py
	$(PYTHON) gpu/metal/embed_source.py

kmeans_metal: gpu/metal/kmeans_metal.mm gpu/metal/kernels_source.h kmeans_common.hpp
	$(CXX) $(BASE_FLAGS) $(WARN_FLAGS) $(METAL_FLAGS) -I. $< $(METAL_LIBS) -o $@

test: all
	$(PYTHON) tests/run_tests.py
	$(PYTHON) tests/run_resampled_tests.py
	$(PYTHON) tests/run_weak_scaling_tests.py
	-$(PYTHON) tests/run_metal_tests.py

smoke: all
	$(PYTHON) scripts/smoke_test.py

clean:
	rm -f kmeans_serial kmeans_openmp kmeans_mpi kmeans_hybrid kmeans_metal
	rm -f gpu/metal/kernels_source.h
