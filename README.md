# Covertype K-means: serial and OpenMP baselines

`kmeans_serial.cpp` implements single-threaded Lloyd K-means on UCI Covertype.
The input is the original `covtype.data` CSV: 54 features followed by the
known class label, which is excluded from clustering. The first 10 continuous
features are z-scored using population standard deviation; the remaining 44
binary features remain 0/1. Initialization uses k-means++ with a fixed seed.
Empty clusters retain their previous centroid. Ties go to the lowest cluster ID.
The loop stops when labels do not change or `MAX_ITER` is reached.

Compile with an optimizing C++17 compiler, for example:

```sh
g++ -O3 -std=c++17 -DNDEBUG kmeans_serial.cpp -o kmeans_serial
./kmeans_serial data/covtype.data 7 100 42
```

On this Windows machine, Zig 0.13 is installed in `.venv` and can compile it:

```powershell
& .\.venv\Lib\site-packages\ziglang\zig.exe c++ -O3 -std=c++17 -DNDEBUG kmeans_serial.cpp -o kmeans_serial.exe
& .\kmeans_serial.exe .\data\covtype.data 7 100 42
```

Arguments are `DATA.csv [K=7] [MAX_ITER=100] [SEED=42]`.
`fit_s` measures only Lloyd iterations, including assignment and centroid
updates. `init_s` covers k-means++ initialization; `load_s`, `preprocess_s`,
and `validation_s` are separately reported. `inertia` is independently
recomputed using the final centroids. For scaling comparisons, use the same
input, preprocessing, initialization, stopping rule, compiler flags, and node.
Run multiple trials and report the median `fit_s`. Final required timing
figures must be measured on Atlas or Vanda.

Initial local Windows measurement (Zig 0.13, `-O3`, K=7, max 100, seed 42):
581,012 rows, 91 iterations, final inertia 3,712,497.960091. Three `fit_s`
measurements: 16.056455, 13.378046, 14.647283 seconds; median 14.647283 s.
This is a development baseline, not an Atlas/Vanda figure.

## OpenMP version

`kmeans_openmp.cpp` preserves the same input, preprocessing, k-means++ seed,
lowest-ID tie breaking, empty-cluster policy, and label-stability stopping
rule. Points are independent during assignment because centroids are
read-only for the duration of an iteration. Direct writes to shared cluster
sums/counts would race, so each thread accumulates into private arrays.
After the parallel region's barrier, one thread combines arrays in thread-ID
order and updates centroids. Data loading, preprocessing, initialization,
centroid update, and final SSE validation remain serial. Floating-point
summation order can change with the number of threads.

On GCC (including Atlas/Vanda), build **both** versions with matching release
flags, adding `-fopenmp` for OpenMP:

```sh
g++ -O3 -march=native -std=c++17 -DNDEBUG kmeans_serial.cpp -o kmeans_serial
g++ -O3 -march=native -std=c++17 -DNDEBUG -fopenmp kmeans_openmp.cpp -o kmeans_openmp
./kmeans_serial data/covtype.data 7 100 42
OMP_NUM_THREADS=4 ./kmeans_openmp data/covtype.data 7 100 42
```

On Windows, a GCC with OpenMP is required; the project `.venv` Zig compiler
does not include `omp.h`. Set `OMP_NUM_THREADS` to `1`, `2`, `4`, `8`, `16`,
or the available logical processor count. Every program emits the original
human-readable metrics plus `label_hash`, `csv_header`, and `csv` lines.
`fit_s` is the main benchmark metric and excludes reading, preprocessing,
initialization, and final SSE validation. `label_hash` compares the full
label sequence, not just cluster sizes.

Local Windows GCC 12.2.0, `-O3 -march=native`, three runs per setting,
median `fit_s` (seconds), same 581,012 rows, K=7, seed 42:

| Implementation | Threads | Median fit_s | Speedup vs GCC serial | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Serial | 1 | 2.394487 | 1.00 | 100% |
| OpenMP | 1 | 2.725991 | 0.88 | 87.8% |
| OpenMP | 2 | 1.389308 | 1.72 | 86.2% |
| OpenMP | 4 | 0.720191 | 3.32 | 83.1% |
| OpenMP | 8 | 0.450752 | 5.31 | 66.4% |
| OpenMP | 16 | 0.369735 | 6.48 | 40.5% |
| OpenMP | 24 | 0.331562 | 7.22 | 30.1% |

All runs converged in 46 iterations with SSE 3722934.178255 and identical
label hash 12199449322292579472. The old Zig-compiled result had a
different final SSE and iteration count; the C++ standard library's
random-distribution implementation can differ across compilers, even with
the same seed. For valid scaling, compare builds from the same compiler.
Scaling noticeably flattens after 8 threads; barriers/reduction each
iteration and memory bandwidth are plausible contributors. These local
measurements are exploratory. Repeat the experiment on Atlas/Vanda for the
assignment report.
