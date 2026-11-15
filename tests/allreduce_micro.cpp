// Isolates the collective cost of exactly the payload our K-means sends:
// K*D doubles (3024 B) + (K+1) uint64 per rank per iteration.
// Purpose: separate "the fabric is slow" from "the ranks were not scheduled",
// by running the identical communication with and without core binding, and with
// and without OpenMP threads competing for the same cores.
#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int K = 7, D = 54, iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    double* send = (double*)std::malloc(sizeof(double) * K * D);
    double* recv = (double*)std::malloc(sizeof(double) * K * D);
    unsigned long long* cs = (unsigned long long*)std::calloc(K + 1, sizeof(unsigned long long));
    unsigned long long* cr = (unsigned long long*)std::malloc(sizeof(unsigned long long) * (K + 1));
    for (int i = 0; i < K * D; ++i) send[i] = 1.0 / (i + 1);

    double per_iter_ms = 0.0;
    volatile int done = 0;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (tid == 0) {
            for (int r = 0; r < 5; ++r) {
                MPI_Allreduce(send, recv, K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(cs, cr, K + 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            for (int r = 0; r < iters; ++r) {
                MPI_Allreduce(send, recv, K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(cs, cr, K + 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            }
            const double t1 = MPI_Wtime();
            per_iter_ms = (t1 - t0) / iters * 1000.0;
            done = 1;
        } else {
            // emulate the non-MPI threads of a hybrid rank competing for CPU
            while (!done) { }
        }
    }

    double maxv = 0.0, minv = 0.0;
    MPI_Reduce(&per_iter_ms, &maxv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&per_iter_ms, &minv, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("  ranks=%3d threads/rank=%2d  allreduce_pair_ms  median_rank=%8.4f  max=%8.4f  min=%8.4f\n",
                    size, (argc > 2) ? std::atoi(argv[2]) : 1, per_iter_ms, maxv, minv);
    }
    std::free(send); std::free(recv); std::free(cs); std::free(cr);
    MPI_Finalize();
    return 0;
}
