#define main kmeans_mpi_program_main
#include "../kmeans_mpi.cpp"
#undef main

#include <iostream>

int main() {
    constexpr int k = 2;
    std::vector<double> centroids(size_t(k) * D);
    std::vector<double> sums(size_t(k) * D, 0.0);
    std::vector<uint64_t> counts(k + 1, 0);

    for (int j = 0; j < D; ++j) {
        centroids[j] = 100.0 + j;
        centroids[size_t(D) + j] = -100.0 - j;
        sums[size_t(D) + j] = 10.0 + 2.0 * j;
    }
    counts[1] = 2;

    const std::vector<double> previous = centroids;
    update_centroids(centroids, sums, counts, k);

    for (int j = 0; j < D; ++j) {
        if (centroids[j] != previous[j]) {
            std::cerr << "empty cluster centroid changed at feature " << j << '\n';
            return 1;
        }
        const double expected = (10.0 + 2.0 * j) / 2.0;
        if (centroids[size_t(D) + j] != expected) {
            std::cerr << "nonempty cluster mean is wrong at feature " << j << '\n';
            return 1;
        }
    }
    std::cout << "empty cluster retained previous centroid; nonempty cluster updated\n";
    return 0;
}
