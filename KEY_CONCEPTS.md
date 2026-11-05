# Key Concepts — DSA5208 Project 3

## Covertype K-means: serial, OpenMP, MPI, and hybrid MPI + OpenMP

This document discusses the concepts that the reference slides (Lectures 8, 9, 11,
12) and `Project Requirement.txt` expect the final report to demonstrate, and
shows exactly where and how each concept is used in this repository. It is
written to be pasted into, or used as the outline for, the "introduction",
"design decisions", "scaling", and "validation" sections of the PDF report.

Sources:

- `reference_slides/Project Requirement.txt` — the formal deliverables.
- Lecture 8 (MPI), Lecture 9 (OpenMP), Lecture 11 (Kokkos / performance
  portability), Lecture 12 (Design of Parallel Algorithms and the Project 3
  grading rubric).

---

## 0. Executive summary

The project parallelises Lloyd's K-means on the UCI Covertype dataset in four
ways: a serial reference, OpenMP (shared memory), MPI (distributed memory), and
a hybrid MPI + OpenMP version. The central technical difficulty is that K-means
is **not embarrassingly parallel**: distance evaluation and local accumulation
are independent per observation, but every centroid depends on *all* observations
of its cluster, including those owned by other ranks. The parallel design
therefore alternates between local data-parallel work and global reductions every
iteration.

The key concepts exercised, and the lecture each comes from, are:

| # | Concept | Lecture | Where it appears in this project |
|---|---------|---------|----------------------------------|
| 1 | Data vs task parallelism | L12 | Data parallelism over rows; task parallelism only for rank-0 I/O/init |
| 2 | Domain decomposition | L12 | Contiguous row blocks per MPI rank, equal within one row |
| 3 | Shared vs message-passing models | L12 | OpenMP for intra-node, MPI for inter-node |
| 4 | MPI ranks / communicators (SPMD is standard MPI terminology) | L8 | `MPI_COMM_WORLD`, every rank runs the same loop |
| 5 | MPI collectives | L8 | `MPI_Bcast`, `MPI_Scatterv`, `MPI_Allreduce`, `MPI_Finalize`, `MPI_Gatherv` |
| 6 | Blocking and deadlock avoidance | L8 | No point-to-point; all ranks call the same collectives in the same order |
| 7 | Serial initialisation + broadcast | L8 | K-means++ on rank 0, then `MPI_Bcast` of centroids |
| 8 | Reduction / all-reduction | L8 | Cluster sums, counts, changed labels, inertia, timings |
| 9 | MPI communicators (single communicator) | L8 | Whole job in `MPI_COMM_WORLD`; no groups/splits needed |
| 10 | Remote memory access (considered, rejected) | L8 | Not used: the uniform barrier every iteration makes RMA pointless here |
| 11 | Floating-point non-associativity | L8 | Bit-level differences across rank/thread counts; tolerance-based validation |
| 12 | OpenMP threads / parallel regions | L9 | One team per rank, `#pragma omp parallel` |
| 13 | Worksharing and scheduling | L9 | `#pragma omp for schedule(static)` over local rows |
| 14 | Data environment (shared/private) | L9 | Read-only shared centroids/data; per-thread accumulator arrays |
| 15 | Races, atomics, critical sections | L9 | Avoided entirely by private totals + fixed-order merge |
| 16 | Implicit barrier and master/single | L9 | End-of-region barrier; `#pragma omp single` measures team size |
| 17 | OpenMP runtime control | L9 | `OMP_NUM_THREADS`, `omp_set_dynamic(0)`, `omp_get_max_threads`, `omp_get_num_threads` |
| 18 | MPI + X hybrid model | L8/L9 | `MPI_Init_thread(MPI_THREAD_FUNNELED)`; two-level reduction |
| 19 | Hierarchical parallelism | L11 | Node (MPI) → cores (OpenMP) → vectorised inner loop |
| 20 | Communication cost model and arithmetic intensity | L12 | Per-iteration payload is constant `K·D`; call count and latency dominate at scale |
| 21 | Strong/weak scaling, speedup, efficiency (L12 rubric); Amdahl and Gustafson (standard theory) | L12 + standard | Measurement plan in §7; serial rank-0 phases are the Amdahl bottleneck |
| 22 | Validation methodology | L12 | Cross-implementation exact/tolerance comparison, `label_hash`, edge cases |
| 23 | Real-data analysis | Requirement / L12 | Covertype 581,012 × 54; K=7 chosen to match 7 cover types |
| 24 | AI-usage declaration | L12 | Required section, template in §11 |

---

## 1. The algorithm: Lloyd's K-means and why it is the right subject

### 1.1 Algorithm

The implementations compute the same fixed-point iteration:

1. **Initialise** K centroids with K-means++ (seeded `std::mt19937_64`).
2. **Assign** each observation `x_i` to the centroid with the smallest squared
   Euclidean distance; ties go to the lowest cluster ID.
3. **Update** each centroid to the arithmetic mean of the observations assigned
   to it; an empty cluster keeps its previous centroid.
4. **Repeat** until the number of observations whose label changed is zero, or
   until `MAX_ITER`.

`inertia` (called `sse` in the CSV output) is the sum of squared distances from
each point to its assigned centroid; it is computed once at the end.

Core terms, defined because Lecture 12 explicitly requires all non-standard
terminology to be defined:

- **Centroid** — the mean feature vector of a cluster.
- **Assignment / label** — the cluster index chosen for an observation.
- **Inertia / SSE** — within-cluster sum of squared errors; the objective K-means
  locally minimises. Lower is better, but it always falls as K grows.
- **Convergence** — here, a whole iteration in which no assignment changed.
- **K-means++** — a seeding heuristic that samples each next centroid with
  probability proportional to its squared distance from the current nearest
  centroid; it reduces poor local minima compared with uniform seeding.

### 1.2 The dataset

`data/covtype.data` is the UCI Forest CoverType dataset:

- **581,012 rows**, **54 numerical features**: 10 continuous features (elevation,
  slope, distances to water/roads/fire points, hillshade at three times of day)
  and 44 binary indicator features (wilderness areas and soil types).
- Column 55 is the known forest cover type (7 classes). It is **excluded from the
  distance computation** (`kmeans_*.cpp` read and discard it), and is used only
  for the data conclusions in §10.
- **Preprocessing**: the 10 continuous columns are z-scored with an online
  mean/M2 (Welford) pass and population standard deviation; a zero-variance
  column maps to 0. The 44 binary columns are left untouched because 0/1 is
  already a comparable scale — z-scoring them would give a single rare indicator
  more weight than all the physical distances.
- **K = 7 by default**, matching the number of cover types, so the clustering can
  be compared with the labels even though K-means never sees them.

The preprocessing choice matters for the parallelisation: it is a column-wise
operation over the full matrix, done once, before distribution. It is therefore
serial and belongs to the Amdahl analysis (§7.3), not the parallel loop.

### 1.3 Why K-means is not embarrassingly parallel

Lecture 12 separates **data parallelism** (split the data, run the same
instructions) from **task parallelism** (different processes run different code).
K-means looks data-parallel: each point's nearest-centroid search needs only
that point and the centroids. If the algorithm stopped there it would be
embarrassingly parallel.

It does not stop there. The update step is a **reduction over the data**: a
centroid is the mean of every point in its cluster, and after domain
decomposition the points of one cluster are spread over several ranks. Therefore:

- the per-cluster **sum vectors** and **counts** must be combined across ranks
  every iteration;
- the **convergence flag** (number of changed labels) is also global;
- every rank must observe the new centroids before the next assignment step.

So each iteration is a strictly alternating sequence

```
local data-parallel work  →  global reduction  →  barrier  →  replicated update
```

and the reduction/barrier is on the critical path. This is the "not
embarrassingly parallel" argument the requirement asks for, and it is the reason
the project has meaningful communication to analyse.

---

## 2. The four implementations and the decomposition

| Binary | Model | Parallelism | Notes |
|--------|-------|-------------|-------|
| `kmeans_serial` | reference | none | Loads, preprocesses, initialises, fits, validates |
| `kmeans_openmp` | shared memory | OpenMP threads over rows | One process; all data resident |
| `kmeans_mpi` | message passing | MPI ranks over row blocks | Rank 0 loads and scatters |
| `kmeans_hybrid` | MPI + OpenMP | ranks × threads | Two-level reduction |

### 2.1 Decomposition

Lecture 12's **domain decomposition** pattern is used: the observation matrix is
split into contiguous row blocks, which is the natural 1-D decomposition because
the only coupling between rows is the per-cluster sum.

```
base      = rows / ranks
remainder = rows % ranks
local_rows(rank) = base + (rank < remainder ? 1 : 0)
```

The remainder rows are given to the lowest ranks so the blocks stay contiguous
and differ by at most one row. The block layout is computed identically on every
rank from `rows` and `ranks`, so no per-rank metadata has to be communicated
beyond the row count itself. This is a deliberate design choice: it avoids a
second collective to agree on counts and displacements.

After the scatter, rank 0 **releases** its full copy of the matrix
(`std::vector<double>().swap(x)`), so peak memory is full data on rank 0 during
load and `n/p` rows per rank afterwards.

### 2.2 Why rows, not features

Splitting rows keeps the per-iteration reduction size independent of the data
size: each rank contributes a `K × 54` partial-sum matrix and `K` counts, no
matter how many rows it owns. Splitting features would instead require every rank
to evaluate all `K` centroids on all rows and then reduce distances — more
communication and worse cache behaviour.

---

## 3. MPI concepts (Lecture 8)

### 3.1 Ranks, communicators, SPMD

An MPI program is a set of independent processes (**ranks**). This project uses
the **SPMD** style (standard MPI terminology, not a slide term): every rank
executes the same `run()` function and branches
only for I/O, initialisation, and printing. All communication is inside
`MPI_COMM_WORLD`. Lecture 8 describes communicators as an ID plus a rank list;
because the group is fixed for the whole run, there is no need for
`MPI_Comm_split` or MPI groups here. That is a design decision worth stating:
splitting communicators pays off when different subsets of ranks must
communicate among themselves, which never happens in this algorithm.

### 3.2 The collective sequence

Every rank executes exactly the same sequence of collectives. This is what makes
the program deadlock-free without any message tags or point-to-point matching
logic.

1. `MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD)` — broadcast the row count
   (needed by every rank to compute its block).
2. `MPI_Scatterv` — send each rank its contiguous row block. `Scatterv` is used
   rather than `Scatter` because the block sizes differ by one row when the row
   count is not divisible by the rank count. This directly implements the
   varying-size scatter described in Lecture 8.
3. `MPI_Bcast(centroids, K*D, MPI_DOUBLE, 0, ...)` — broadcast the initial
   centroids computed by K-means++ on rank 0. This is the lecture's canonical
   pattern "broadcast after serial initialisation": it costs one collective and
   guarantees every rank starts from **bitwise identical** centroids, which is a
   precondition for comparing runs later.
4. Inside every iteration:
   - `MPI_Allreduce(sums, ..., K*D, MPI_DOUBLE, MPI_SUM, ...)` — combine the
     per-cluster coordinate sums.
   - `MPI_Allreduce(count_stats, ..., K+1, MPI_UINT64_T, MPI_SUM, ...)` — combine
     the cluster counts **and** the local changed-label count in one call
     (`count_stats[K]` holds the changed-label total).
5. After fitting: `MPI_Reduce(local_sse, ..., MPI_SUM, 0, ...)` for inertia and
   `MPI_Gatherv` for the final labels, so rank 0 can recompute the serial label
   hash in the original row order.
6. Timing values are reduced with `MPI_Reduce(..., MPI_MAX, 0, ...)` so the
   report can state the slowest rank's wall time, not an average that hides
   stragglers.

### 3.3 Reduce vs Allreduce — a real design decision

`MPI_Reduce` leaves the result on the root; `MPI_Allreduce` leaves it on every
rank. There are two legitimate designs:

- **Reduce + Broadcast**: root computes the centroids, then broadcasts `K·D`
  doubles. Cost: reduce + broadcast.
- **Allreduce + replicated update**: every rank receives the sums and counts and
  each performs the same `K·D` divisions locally. Cost: allreduce only.

The project chooses the second. The redundant centroid update is only `K·D = 378`
divisions, whereas a broadcast is another full collective; since the two-level
reduction is latency-dominated (Lecture 12's cost model, §6), removing a
collective call per iteration is the better trade. This is a concrete example of
"avoiding communication" by preferring fewer communication calls over fewer
floating-point operations.

### 3.4 Allreduce vs Gather — where labels live

Assignments are produced locally every iteration, but the algorithm only needs
the **number** of changed assignments, which is a scalar reduction. Labels
themselves are not needed by other ranks until the run ends. Therefore:

- labels are never exchanged during the loop (a `Gather`/`Allgather` per
  iteration would add `O(n/p)` data movement and another collective);
- after the loop, one `MPI_Gatherv` collects the labels in rank order so rank 0
  can compute the same `label_hash` as the serial program.

This is the single most important communication-avoidance decision in the
project. It is exactly the reasoning in Lecture 12: reduce the number of
communication calls and keep payloads off the critical path.

### 3.5 Blocking semantics, deadlock, and why no point-to-point

Lecture 8 stresses that blocking sends/receives can deadlock and hurt efficiency.
This project avoids the issue by construction:

- there are **no** `MPI_Send`/`MPI_Recv` calls at all, only collectives;
- every rank calls the collectives in the same order with the same arguments, so
  matching is never ambiguous;
- collectives are themselves blocking (they return only when the collective
  completes), but they act as the necessary synchronisation point around which
  the centroid update is defined.

The slide's asynchronous variants (`MPI_Isend`, `MPI_Ssend`, `MPI_Bsend`) and
non-blocking collectives are not needed because the algorithm has a genuine global
synchronisation every iteration; overlapping the reduction with useful work would
require speculative computation against centroids that do not exist yet.

### 3.6 Remote memory access (RMA), considered and rejected

Lecture 8 presents RMA (`MPI_Win_create`, `MPI_Put`, `MPI_Get`,
`MPI_Accumulate`, fences and locks) as the third interaction model, useful when
synchronisation is difficult. Here synchronisation is not difficult: the
algorithm has a hard barrier each iteration because the next assignment cannot
start until every centroid is final. RMA would replace a collective with a set of
one-sided puts plus a fence, i.e. the same synchronisation with more moving parts
and no reduction primitive, so it is not used. Being able to **argue** this is
part of the design discussion: choosing the simplest communication pattern that
matches the algorithm.

If the project were extended, RMA or `MPI_Accumulate` on a window of cluster sums
would be the natural way to overlap partial sums from different ranks, at the
cost of more complex synchronisation.

### 3.7 Floating point does not associate (Lecture 8)

IEEE-754 addition is not associative: `(a+b)+c ≠ a+(b+c)` in general. MPI's
reductions combine contributions in an implementation-defined tree, so changing
the rank count changes the order of additions and therefore low-order bits of the
cluster sums, centroids, and eventually distances. Lecture 8 makes exactly this
point and notes that reproducibility is itself a possible project topic.

In this project the effect is measurable and is documented rather than hidden:

- the small deterministic dataset matched across implementations with a maximum
  centroid difference of `1.110e-15`;
- real Covertype (581,012 rows, K=7, 5 fixed iterations) matched with a maximum
  centroid difference of `2.706e-13`;
- the test harness compares floating-point quantities with tolerances and
  compares discrete quantities (iteration count, converged flag, cluster sizes,
  label hash) exactly.

Three mitigations keep the comparison meaningful:

1. **Fixed starting point** — K-means++ runs once on rank 0 and is broadcast, so
   all implementations start from the same centroids.
2. **Deterministic local reductions** — within a rank, OpenMP thread totals are
   merged in fixed thread-ID order, so thread count changes, not scheduling,
   determine rounding.
3. **`--fixed-iterations` mode** — for performance measurements, running a fixed
   number of iterations avoids the situation where a single flipped near-tie
   changes the convergence iteration and therefore the timing.

The residual risk is a point whose two nearest centroids are so close that a
`1e-13` perturbation flips its label; the test suite's tolerance policy and fixed
iteration mode make this visible rather than silently changing results.

---

## 4. OpenMP concepts (Lecture 9)

### 4.1 Threads, parallel regions, and the fork-join model

OpenMP is a shared-memory, directive-based model: the programmer declares where
parallelism happens and the compiler generates the thread management. In
`kmeans_openmp.cpp` the structure is the textbook fork-join loop:

```cpp
#pragma omp parallel
{
    const int tid = omp_get_thread_num();
    double* thread_sums = &local_sums[size_t(tid) * k * D];
    ...
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        ... // nearest-centroid search, label update, accumulation
    }
}   // implicit barrier
```

Lecture 9's four requirements for a thread model map directly:

- **create/destroy threads** — the parallel region does it;
- **distribute computation** — the `omp for` worksharing construct does it;
- **coordinate threads** — the implicit end-of-region barrier does it;
- **shared vs private memory** — OpenMP's default data-sharing rules do it
  (no explicit `shared`/`private` clauses are needed here).

### 4.2 Worksharing and `schedule(static)`

Lecture 9 contrasts `static` (split at compile time, no scheduling overhead),
`dynamic` (assign at run time, more overhead but better for irregular work), and
`guided` (a middle ground).

K-means assignment is perfectly regular: every row costs the same `K·D` distance
operations. Static scheduling is therefore the right choice — it has no runtime
overhead, gives contiguous row blocks with good cache/stride behaviour, and is
deterministic. Dynamic scheduling would add a synchronisation point per chunk for
no load-balancing benefit. This is a small but explicit design decision to defend
in the report.

### 4.3 The data environment: shared data, private accumulation

Lecture 9 warns that unintended sharing causes race conditions and that
synchronisation is expensive; the recommended fix is to change the memory access
pattern rather than add locks.

The project applies that advice literally:

- `x`, `centroids`, and `labels` are **shared**;
- each point's `labels[i]` has exactly one writer (the thread that owns row `i`),
  so labels never race;
- the global `sums` and `counts` arrays are **not** updated by threads. Instead
  each thread writes into its own slice of `local_sums` and `local_counts`,
  indexed by `tid`. No atomics and no `critical` sections appear anywhere in the
  fitting loop.
- Only after the implicit barrier does the main thread merge the thread slices in
  fixed `tid` order into the shared totals.

This is race-free by construction (data partitioning instead of locking), and it
is deterministic for a fixed thread count. The cost is `T × K × 54` doubles of
extra memory per rank, which is negligible (`4 threads × 7 × 54 × 8 B ≈ 12 KB`).

### 4.4 Synchronisation primitives

Lecture 9 lists `critical`, `atomic`, `barrier`, `ordered`, locks, and the
`master`/`single` constructs. The project uses only two of them:

- the **implicit barrier** at the end of the parallel region (no explicit
  `#pragma omp barrier` is needed) guarantees all thread totals are visible
  before the merge;
- `#pragma omp single` in the hybrid version runs one call to
  `omp_get_num_threads()` so the program records the **actual** team size rather
  than the requested maximum.

No `atomic` or `critical` is used because no two threads ever write the same
location. Reporting that fact is stronger than saying "we used atomics": it shows
the reduction was designed, not patched.

### 4.5 Runtime control and thread counts

- `OMP_NUM_THREADS` sets the team size; `omp_get_max_threads()` reports it.
- the hybrid version calls `omp_set_dynamic(0)` so the runtime cannot silently
  shrink the team, then calls `omp_set_num_threads(threads)` with the measured
  actual size. Dynamic teams would make reported `threads` and the per-thread
  array indexing inconsistent.
- the hybrid version additionally checks, with `MPI_Allreduce(..., MPI_MIN)` and
  `MPI_MAX`, that **all ranks ended up with the same thread count**, and aborts
  otherwise. A rank with fewer threads would produce a valid but differently
  shaped reduction, and the run would no longer be a controlled experiment.

### 4.6 Memory bandwidth is the intra-node limit

K-means assignment is a low-arithmetic-intensity kernel: each point reads `D = 54`
doubles (432 B) and performs roughly `K·D·3 ≈ 1,000` flops, i.e. a few flops per
byte. Lecture 11's data-layout discussion and Lecture 9's caching discussion both
point the same way: the loop is limited by how fast the row data can be streamed
from memory, not by the floating-point units.

Consequences for the analysis:

- adding OpenMP threads gives near-linear speedup only while the memory
  subsystem has spare bandwidth; beyond that, speedup saturates;
- the loop is already cache-friendly: rows are contiguous, the centroids
  (`K·D·8 = 3 KB`) stay in L1/L2, and the inner `j` loop is vectorisable and
  stride-1;
- this is why the reported configuration should include the number of cores per
  node and the rank × thread layout — the same total worker count can give
  different bandwidth per rank.

---

## 5. The hybrid model: MPI + OpenMP (Lecture 8's "MPI + X")

### 5.1 Why combine them

Lecture 8 describes MPI+X as the standard pattern: MPI across pools of memory
(nodes), X inside a pool (cores). Lecture 9 adds the concrete question for this
project: *what do you parallelise with MPI and what with OpenMP?*

The answer used here:

- **MPI** distributes rows across nodes. This is the only way to use more memory
  than one node, and it is the level where the algorithm's global reduction must
  happen anyway.
- **OpenMP** distributes a rank's local rows across cores on the same node.
  Threads share the rank's memory, so no data has to be copied and no message is
  needed for the thread-level reduction.

A pure-MPI run on one node can use the same cores, but it pays for `p` separate
copies of the centroid array and `p` messages where one thread team needs none.
A hybrid run can also match the hardware topology exactly (one rank per socket
or per NUMA domain, one thread per core), which is the configuration Lecture 9
asks the student to experiment with: on a 72-core Vanda node, is 1 rank × 72
threads, 2 × 36, or 4 × 18 fastest?

### 5.2 Thread support level

The hybrid uses:

```cpp
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
if (provided < MPI_THREAD_FUNNELED) { /* reject and abort */ }
```

`MPI_THREAD_FUNNELED` means "only the thread that called `MPI_Init_thread` may
make MPI calls". That is exactly the usage pattern: all MPI calls are made by the
main thread **outside** any parallel region; OpenMP threads only touch shared
memory. This is the cheapest sufficient thread level (it lets the MPI library
skip per-call locking), and it directly answers Lecture 9's question "how do
threads interact with MPI windows?" — here they do not, because there are no
windows and no OpenMP worker thread calls MPI — the main thread makes every MPI
call.

The program also re-queries the level with `MPI_Query_thread()` and refuses to
run if support fell below `MPI_THREAD_FUNNELED`, so a misconfigured MPI build
fails loudly instead of corrupting results.

### 5.3 The two-level reduction

This is the conceptual heart of the hybrid design and the cleanest thing to
explain in the report:

```
Level 1 (shared memory):  per-thread partial sums  --fixed order-->  per-rank totals
Level 2 (message passing): per-rank totals         --MPI_Allreduce--> global totals
```

- Level 1 is a **sequential merge** of `T` thread slices in fixed thread-ID
  order. It costs `T·K·54` additions, runs on one thread, and is deterministic.
- Level 2 is a **single `MPI_Allreduce`** of `K·D` doubles plus one of `K+1`
  integers. It is deterministic only up to floating-point ordering, as discussed
  in §3.7.

This hierarchy mirrors the hardware hierarchy (Lecture 11's hierarchical
parallelism: node → cores → vector units). It also means the number of messages
per iteration is **independent of the number of threads**: going from 4×1 to
1×4 changes only the Level-1 merge size, not the MPI payload. That is a useful
prediction to test in the scaling study.

### 5.4 Layouts to measure

The expected configurations are `1×4`, `2×2`, `4×1`, and `2×4`/`4×4` when
enough cores exist. The interesting question is not whether hybrid beats pure MPI
in every case — often it does not at small core counts on one node — but **where
the crossover is**, and whether the best layout tracks the NUMA topology. This is
exactly the trade-off the rubric rewards: fewer ranks means fewer, larger
messages; more ranks means more latency but more independent memory streams.

---

## 6. Communication cost and arithmetic intensity (Lecture 12)

### 6.1 The three-term cost model

Lecture 12 gives the runtime of a parallel code as

```
T = (flops × t_flop) + (communication volume / bandwidth) + (number of calls × latency)
```

and notes the gap between these terms is growing: time per flop improves ~59% per
year, bandwidth ~26%, latency ~15%. In other words, **latency is becoming
relatively more expensive**, so reducing the number of communication calls
matters more than reducing bytes.

This project applies the model as follows. Per iteration, per rank:

- **flops**: `local_rows × K × D × ~3` (distance evaluation) plus `local_rows × D`
  (accumulation). This scales as `n/p`.
- **volume**: `K·D` doubles = `7×54×8 = 3,024 B` for the sum reduction, plus
  `(K+1)×8 = 64 B` for the count/changed reduction. Crucially, this is
  **independent of `n`**.
- **calls**: 2 all-reductions per iteration. That is the number the slides tell
  you to minimise, and the implementation already merges counts and changed
  labels into one call to keep it at two.

So as the problem grows at fixed `p`, the compute term grows while volume and
call count stay constant: K-means has **good arithmetic intensity at scale**. As
`p` grows at fixed `n`, compute per rank shrinks but the collective cost stays
constant, so the communication fraction rises and strong scaling must eventually
flatten. Making that argument, and then measuring it with `mpi_reduction_s` and
`fit_s`, is the requested "impact of communication on the runtime".

### 6.2 What the code measures

To support the analysis, the MPI builds report components separately:

| Field | Meaning |
|-------|---------|
| `fit_s` | wall time of the whole Lloyd loop (primary metric) |
| `local_compute_s` | cumulative rank-local assignment + accumulation (incl. thread merge) |
| `mpi_reduction_s` | cumulative time inside the per-iteration all-reductions |
| `distribution_s` | time from the start of distribution (the row-count broadcast) through the row scatter and centroid broadcast |
| `load_s`, `preprocess_s`, `init_s` | rank-0-only serial phases |
| `validation_s` | final inertia, inertia reduction, label gather |
| `total_s` | max rank wall time from `run()` entry through validation |

All cross-rank timings are reduced with `MPI_MAX`, so they are the slowest rank's
values; the report should state that definition rather than assuming the
components sum exactly to `fit_s` (barriers, waits, and timing calls are not
inside any component).

### 6.3 Red-blue pebbling and lower bounds — why they are background here

Lecture 12 also develops red-blue pebbling and Loomis-Whitney bounds for
communication-optimal algorithms (matrix multiplication, stencils). Those tools
assume structured data access and a cache hierarchy; K-means' communication is a
fixed `K×D` reduction per iteration, so the useful model is the three-term cost
model above rather than a pebbling lower bound. Mentioning that distinction
shows the student knows when the theory applies.

---

## 7. Scaling and efficiency: the measurement plan (Lecture 12 rubric)

The Project 3 grading notes define the analysis required: strong and weak
scaling, speedup, efficiency, and the communication impact. Amdahl's and
Gustafson's laws are not in the slides but are the standard tools for
interpreting those numbers, so they are added here. Treat the section as a
checklist:

### 7.1 Serial scaling with problem size (5 points)

Vary the problem size for the serial reference and plot runtime vs size on
**logarithmic axes** (the slides are explicit about log axes for time, problem
size, and CPU count). For this dataset the natural knob is the number of rows
(subsample 50k, 100k, 200k, 400k, 581k) and optionally K. Confirm the expected
`O(n·K·D·iterations)` shape and report the fitted exponent.

Do not vary only `n` at fixed K if the point is to characterise the algorithm;
varying K as well separates the per-iteration cost from the number of iterations.

### 7.2 MPI strong and weak scaling (10 points)

- **Strong scaling**: fix `n` (full Covertype) and increase the number of ranks
  (1, 2, 4, 8, ..., at least 256 cores on Atlas). Report
  `S(p) = T(1)/T(p)` and `E(p) = S(p)/p`, plus the communication fraction
  `mpi_reduction_s / fit_s`. Log-log plot.
- **Weak scaling**: keep rows per rank constant while increasing `p`, so the
  ideal runtime is flat. Report efficiency against the `p = 1` point. For a real
  dataset the clean way to do this is extrapolation/replication or a synthetic
  generator; Lecture 12 explicitly says synthetic data is an easy way to do weak
  scaling **as long as the generation method is explained** (e.g. `G` Gaussian
  blobs with known means and covariance, generated from a documented seed).
- Minimum target from the rubric: **at least 256 CPU cores** on Atlas. The
  correctness runs in the README are explicitly *not* the scaling study.
- Run repeated trials (3–5) and report medians with spread; MPI timings are noisy
  on shared clusters.

### 7.3 Amdahl's law and the serial fraction

`load`, `preprocess`, and K-means++ `init` all run on rank 0 before any parallel
work. If `f` is the serial fraction of the serial runtime, Amdahl gives the best
possible speedup `S(p) = 1 / (f + (1-f)/p)`, so the serial phases cap strong
scaling no matter how many cores are used.

Two predictions to verify and discuss:

- at 256 ranks, `fit_s` is small, so `load_s + preprocess_s + init_s` dominates
  `total_s` and measured speedup will plateau well below 256;
- the K-means++ initialisation itself is `O(n·K·D)`, about one Lloyd iteration's worth of work, which is why it is kept serial for
  exact reproducibility but must be counted in the serial fraction.

Mitigations worth mentioning (and, if implemented, testing): parallel CSV parsing
or a binary input format, parallel K-means++ sampling, or amortising the load cost
by a longer fit. The honest framing — "the algorithm scales, the I/O and
initialisation do not" — is exactly the kind of conclusion the rubric rewards.

### 7.4 Gustafson's law

Weak scaling keeps the serial load growing with `n` while the parallel part per
rank is fixed, so efficiency still degrades, just more slowly than Amdahl's
constant-problem analysis suggests. State the difference; it shows the two laws
are being applied for their intended purpose.

### 7.5 MPI + X analysis (10 points)

Compare pure MPI against hybrid at the same total worker count (`4×1` vs `2×2` vs
`1×4`). Report:

- `fit_s` and speedup against total workers;
- `local_compute_s` (does the OpenMP merge cost show up?);
- `mpi_reduction_s` (does fewer ranks mean less time in MPI?);
- the best layout, and whether it matches the node's NUMA topology.

Lecture 9's specific questions make good subsection headings: what is the most
effective way to synchronise ranks and threads, how much synchronisation is
needed, and how threads interact with MPI.

### 7.6 The GPU requirement

The rubric says: **if X = OpenMP, you must also test on at least one GPU.** The
current repository is CPU-only, so this is the largest open item (§12). The
assignment loop is the natural offload target: it is a regular `n × K × D` loop
with no dependencies between rows, exactly the form of Lecture 9's

```cpp
#pragma omp target teams distribute parallel for
```

The complication is the accumulation: 54 features × K clusters per point, so
naive atomics would serialise. The scalable options are a per-team reduction, a
segmented/blocked reduction over a grid of `(cluster, feature)` accumulators, or
throwing away per-point savings and doing a two-pass reduction. Discussing that
trade-off is itself worth points, even before the numbers.

---

## 8. Correctness and validation (Lecture 12: "explain your validation tests")

### 8.1 The oracle

The serial program is the oracle. A parallel run is correct if it produces, for
the same input and parameters:

1. the same iteration count and convergence status;
2. the same inertia within floating-point tolerance;
3. the same centroid coordinates within tolerance;
4. the same cluster sizes (exact integers);
5. the same complete label sequence, checked via an order-sensitive hash.

`label_hash` is an FNV-1a-style 64-bit hash over `label+1` in original row
order (the offset basis and prime used in the source), evaluated in row order. Because the labels
are replaced by an order-sensitive hash, any permutation or misalignment caused
by a scatter/gather bug changes the hash. This is stronger than comparing
inertia: inertia can be identical for different assignments.

The harness compares exactly these fields: `rows`, `k`, `max_iter`, `iterations`,
`converged`, `stopping_mode`, `cluster_sizes` and `label_hash` **exactly**;
`inertia` with `rel_tol = 1e-9`, `abs_tol = 1e-4`; and every centroid coordinate
with absolute tolerance `1e-7`. The tolerances are wide enough to absorb
reduction-order rounding (observed `2.7e-13`) and narrow enough that a genuine
algorithmic difference cannot pass.

### 8.2 Why tolerances for floats and exactness for labels

From §3.7: sums over different rank/thread counts round differently, so centroids
can differ in the last bits (`1.1e-15` on the small set, `2.7e-13` on real
Covertype). Requiring bitwise equality would fail essentially every multi-rank run. Labels, in
contrast, are discrete decisions, so they can be compared exactly — unless a
point sits on a decision boundary and the last-bit difference flips it. That is
precisely why the suite also compares iteration count and convergence status, and
why `--fixed-iterations` exists for timing runs.

### 8.3 Edge cases covered

The test harness (`tests/run_tests.py`) drives the binaries as subprocesses,
parses their printed fields, and compares them as above. The fixture is a
deterministic 503-row mixture written from several known blobs (`seed = 5208`);
503 is deliberately not divisible by 3 or 4, and the script asserts that
property. It exercises configurations that specifically stress the parallel
machinery:

- **row counts not divisible by the rank count** — exercises `MPI_Scatterv`
  uneven blocks and `MPI_Gatherv` displacements;
- **more ranks than observations** (3 rows, 4 ranks) — zero-row ranks must still
  join every collective and contribute zeros; a rank that skips a collective
  would hang;
- **K = 1 and K = 7** — degenerate and realistic cluster counts;
- **thread counts 1, 2, 4** — including single-thread runs, which are the closest comparison with the serial
  reference;
- **hybrid `1×4`, `2×2`, `4×1`** (and `2×4` when at least eight CPU threads exist);
- **default convergence and fixed-iteration stopping**;
- **a direct unit test of the empty-cluster path**: a zero-count cluster keeps its
  previous centroid while a non-empty cluster receives its arithmetic mean.

Lecture 12 also warns against meaningless test counts ("if you have 100+ tests,
why?"). The suite is deliberately small and each configuration has a stated
purpose, which is the right way to present it.

### 8.4 Documented results

The README records the executed validation (Apple Clang 21, MPICH 5.0.2,
libomp 18.1.3):

- small K=7 fixed-iteration matrix: maximum centroid difference `1.110e-15`;
- real Covertype, 5 fixed iterations: maximum centroid difference `2.706e-13`;
- full K=7 convergence: 91 iterations, inertia `3712497.960091`, label hash
  `9039984004050467668`;
- empty-cluster unit test passes.

These are **correctness** numbers, not scaling numbers; the report should keep
them clearly separated from the Atlas/Vanda runtime figures.

---

## 9. Performance portability and the choice of X (Lecture 11)

The project requirement allows `MPI + {OpenMP, CUDA, Kokkos, *}`. Lecture 11
frames the choice as **performance portability**: one source that runs well on
many architectures, with execution spaces (where), execution patterns
(`parallel_for`/`reduce`/`scan`), execution policies, memory spaces, layouts, and
traits.

Why OpenMP was chosen:

- Atlas (CPU-only, unlimited use) is the main scaling target, and OpenMP is the
  cheapest way to use all its cores;
- the existing serial code is already a `for` loop over rows, so the rewrite is
  local;
- `-fopenmp` is a compiler flag, not a new dependency, which keeps the build and
  the 256-core scaling study simple.

What Kokkos would have offered, and what it would have cost:

- **pros**: one source for CPU and GPU backends, `Kokkos::View` with explicit
  layouts, `parallel_for`/`parallel_reduce`, team policies and scratch memory for
  hierarchical parallelism — the natural fit for the GPU half of the rubric;
- **cons**: another dependency to build on Atlas/Vanda, and the reduction
  structure would look the same (per-team partial sums + global reduction), so it
  would not remove the algorithmic communication; it would mainly change *where*
  the kernel runs.

What CUDA would have required: a device kernel for assignment, `cudaMalloc`/
`cudaMemcpy` for the rows, and — because the update needs per-cluster sums — a
device-wide reduction or atomics. That is the most work and locks the project to
Nvidia hardware.

Lecture 11's core warning applies to the current implementation: **OpenMP code
that is fast on a CPU is not automatically fast on a GPU** ("using locks on a few
threads on a CPU won't scale well on a GPU"). The CPU design deliberately avoids
atomics; a GPU port cannot simply add them per point without a big slowdown, which
is why §7.6 proposes a structured reduction instead.

---

## 10. Conclusions about the data (requirement: real-world dataset)

The report must discuss what was learned about Covertype, not just about the
parallel code. Points the implementation supports:

- **Scale and shape**: 581,012 rows, 54 features, 7 classes. The full matrix is
  `581,012 × 54 × 8 B ≈ 251 MB` in double precision — small enough for one node,
  which is why strong scaling to 256 cores is a *communication/latency* study
  rather than a memory-capacity study.
- **Feature scaling matters**: the 10 continuous features span very different
  ranges from the 44 binary indicators; z-scoring only the continuous block makes
  Euclidean distance meaningful without inflating rare binary indicators.
- **K = 7 is an assumption**, chosen to match the number of cover types so the
  result can be compared with labels. K-means minimises inertia, not class
  separation, so the clusters need not correspond one-to-one to cover types.
  Quantify this with a contingency table and purity/ARI/NMI computed against
  column 55 (read separately, never used in fitting).
- **Cluster balance**: report `cluster_sizes`. Strongly unequal sizes, and any
  empty clusters (which retain their previous centroid by policy), are evidence
  about the geometry of the data and about the suitability of K.
- **Inertia is not comparable across K**: it decreases monotonically with K, so
  the elbow curve (K = 1..10) is the honest way to discuss the choice of K.
- **Determinism**: the same seed and input give the same final inertia and label
  hash across implementations (up to low-order bits), which makes the data
  conclusions reproducible.

A useful one-line framing for the report: *the parallel implementation is a
scalable way to compute an approximate, geometry-based partition of a real forest
dataset; the clustering itself is only as meaningful as the feature scaling and
the choice of K, which is why the data discussion is separate from the speedup
discussion.*

---

## 11. AI-usage declaration (requirement)

Lecture 12 is blunt: if AI is used it must be declared, and the report must still
make sense with every term defined. A declaration should be specific about
**what** was assisted, **what** the author verified, and **what** the AI did not
produce. Template to adapt truthfully:

> **AI usage.** I used an AI coding assistant (tool/model, dates) for: (i)
> scaffolding the MPI/OpenMP variants from the serial reference; (ii) timing
> instrumentation and output formatting; (iii) drafting the Python validation
> harness; (iv) editorial help with this report. I built and ran every
> implementation myself, reviewed every line of submitted code, re-derived the
> algorithm and the scaling analysis, and collected all experimental results on
> Atlas/Vanda. No experimental numbers in this report were generated by the AI.
> The validation results in §8.4 were reproduced locally with `make && make test`.

Do not claim assistance that did not happen, and do not omit it if it did; the
slides note that undeclared AI use is usually obvious.

---

## 12. Gap analysis against the Project 3 rubric

| Rubric item | Points | Status in repo | What remains |
|---|---|---|---|
| Explain the algorithm | 5 | Done (README §Common algorithm; §1 here) | Condense into report prose |
| Serial scaling vs problem size | 5 | **Missing** | Subsample rows and vary K; log-log plot |
| MPI runtime, ≥256 cores, strong + weak scaling, speedup/efficiency, communication impact | 10 | Code instrumented; only small local runs recorded | Atlas job script, 1→256 ranks, repeated trials, plots |
| MPI+X analysis (OpenMP ⇒ ≥1 GPU test) | 10 | Hybrid implemented and validated on CPU | **GPU test missing** — OpenMP target offload or Kokkos/CUDA kernel |
| Overall discussion incl. data conclusions and parallelisation choices | 5 | This document + README | Integrate, add contingency-table analysis |
| Plots (log axes for time/size/CPU count) | — | **Missing** | Generate from CSV output |
| Runtime figures from Atlas/Vanda | — | **Missing** | Submit jobs, capture `fit_s`, `total_s`, `mpi_reduction_s` |
| Source code + README | — | Done | Ensure `make test` passes on the cluster |
| AI-usage explanation | — | **Missing** | Add §11-style declaration |

The code and the correctness story are in good shape; the remaining work is
almost entirely **measurement and presentation**, plus the GPU half of the MPI+X
requirement.

---

## 13. Glossary

| Term | Definition |
|---|---|
| Rank | One MPI process; has its own memory and an index in a communicator |
| Communicator | A group of ranks with an ID that can communicate collectively (`MPI_COMM_WORLD` here) |
| Collective | An operation involving all ranks of a communicator (`Bcast`, `Scatterv`, `Allreduce`, `Gatherv`) |
| Reduction | Combining many values into one (`MPI_SUM`, `MPI_MAX`) |
| Allreduce | Reduction whose result is available on every rank |
| Domain decomposition | Splitting a data array among ranks and operating on the pieces |
| Owner-computes | The rank that stores a data item is the one that updates it |
| Thread team / parallel region | The set of threads created by `#pragma omp parallel` |
| Worksharing | Distributing loop iterations among the threads of a team |
| Race condition | Two threads access the same location with at least one write and no ordering |
| Implicit barrier | Synchronisation the compiler inserts at the end of a worksharing/parallel construct |
| False sharing | Two threads writing different variables that share a cache line |
| Speedup | `S(p) = T(1) / T(p)` |
| Efficiency | `E(p) = S(p) / p` |
| Strong scaling | Fixed problem size, increasing `p` |
| Weak scaling | Fixed work per rank, increasing `p` |
| Amdahl's law | `S(p) = 1 / (f + (1-f)/p)` for serial fraction `f` |
| Gustafson's law | Scaled-speedup view where the problem grows with `p` |
| Arithmetic intensity | Floating-point operations per byte of communication/memory traffic |
| Inertia / SSE | Sum of squared distances from points to their assigned centroids |
| K-means++ | Distance-weighted seeding heuristic for K-means |
| `label_hash` | FNV-1a-style 64-bit hash of the final label sequence in row order; the exact correctness check |
| Performance portability | One source that performs well across different hardware/compilers |

---

## Appendix: concept → slide → code map

| Concept | Slide source | Code evidence |
|---|---|---|
| MPI rank / communicator | L8 "The MPI Programming Model"; "MPI Communicators" | `kmeans_mpi.cpp:402–425` (`MPI_Init`, `MPI_Comm_rank/size`, `MPI_Finalize`) |
| Broadcast after serial init | L8 "MPI Broadcast" | `kmeans_mpi.cpp:239` (centroid `MPI_Bcast`) |
| Varying-size scatter | L8 "MPI Gather" (`Scatterv`/`Gatherv` variants) | `kmeans_mpi.cpp:227` (`MPI_Scatterv`), `:337` (`MPI_Gatherv`) |
| Reduce / Allreduce | L8 "MPI Reduce", "Reductions" | `kmeans_mpi.cpp:292`, `:300`, `:328` |
| Blocking/deadlock discussion | L8 "Blocking", "Asynchronous Communication" | No point-to-point calls; identical collective sequence on all ranks |
| RMA | L8 "Remote Memory Access in MPI" | Not used; rationale in §3.6 |
| Floating point non-associativity | L8 "Floating Point Arithmetic" | `--fixed-iterations`; tolerance checks in `tests/run_tests.py` |
| Threads / parallel regions | L9 "Threads", "OpenMP Parallel Regions" | `kmeans_openmp.cpp:172`, `kmeans_hybrid.cpp:301` |
| Worksharing / schedule | L9 "Worksharing Constructs", "Scheduling Options" | `#pragma omp for schedule(static)` at `kmeans_openmp.cpp:178`, `kmeans_hybrid.cpp:307` |
| Data environment | L9 "The Data Environment", "Data Clauses" | per-thread `local_sums`/`local_counts`/`local_changed` |
| Avoid races by access pattern | L9 "OpenMP" (unintended sharing) | no `atomic`/`critical`; private slices + fixed-order merge (`kmeans_openmp.cpp:194–206`) |
| Barrier / master / single | L9 "Synchronization", "Master Blocks" | implicit barrier; `#pragma omp single` at `kmeans_hybrid.cpp:255` |
| Runtime routines | L9 "OpenMP Runtime" | `omp_set_dynamic(0)`, `omp_get_max_threads`, `omp_get_num_threads` |
| MPI+X hybrid | L8 intro; L9 "For your projects" | `kmeans_hybrid.cpp:456` (`MPI_Init_thread`), two-level reduction |
| Hierarchical parallelism | L11 "Hierarchical Parallelism" | node → rank → thread → vector loop |
| Views / layouts / traits | L11 "Kokkos Views", "View Layouts", "Memory Traits" | Not used; row-major `std::vector<double>` instead; rationale in §9 |
| Communication cost model | L12 "Actual Impact of Communication" | `mpi_reduction_s` vs `fit_s`; merged count/changed allreduce |
| Avoiding communication | L12 "Avoiding Communication" | labels gathered once; 2 collectives/iteration |
| Data vs task parallelism | L12 "Task Parallelism", "Data Parallelism" | rows are data-parallel; rank-0 I/O/init is the only task-parallel part |
| Domain decomposition | L12 "Domain Decomposition" | row-block formula in `kmeans_mpi.cpp:207–223` |
| Scaling axes / log plots | L12 "Project 3" notes | §7 measurement plan |
| Grading rubric | L12 "Grading" | §12 gap analysis |
| AI declaration | L12 "AI Usage" | §11 |

---

### One-paragraph version

This project parallelises Lloyd's K-means on the Covertype dataset (581,012 × 54,
K = 7) as a case study in a *data-parallel algorithm with a global reduction*.
The row matrix is domain-decomposed across MPI ranks with `MPI_Scatterv`, the
K-means++ initialisation is performed once on rank 0 and broadcast, and every
iteration performs two collective reductions (cluster sums, and counts plus the
changed-label count) so that every rank can apply the same centroid update
locally. Within a rank, OpenMP threads process static row blocks with per-thread
accumulator arrays and a deterministic fixed-order merge, avoiding atomics
entirely. The hybrid version combines the two levels under `MPI_THREAD_FUNNELED`.
Its cost structure — `O(n/p)` local work against a constant `K·D` reduction
payload per iteration — predicts good arithmetic intensity at scale but a rising
communication fraction under strong scaling, a serial-fraction limit from rank-0
loading/preprocessing/initialisation (Amdahl), and a memory-bandwidth ceiling
within a node. Correctness is established against the serial reference through
inertia and centroid tolerances plus an exact, order-sensitive label hash, with
dedicated tests for uneven block sizes, zero-row ranks, empty clusters, and
thread counts from 1 to 4. The remaining report work is measurement (Atlas/Vanda
strong and weak scaling to 256 cores, log-scale plots) and the GPU test required
when the chosen X is OpenMP.
