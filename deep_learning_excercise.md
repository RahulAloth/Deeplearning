Parallelizing Daxpy and Initialization

In this notebook you will parallelize the initialize and daxpy functions to compute the results in parallel using CPUs or GPUs.

/*
// Initialize vectors x and y: parallel algorithm version void initialize(std::vector &x, std::vector &y) { assert(x.size() == y.size());

// Parallelize initialization of `x` using iota view + for_each_n
auto ints = std::views::iota(0);
std::for_each_n(std::execution::par, ints.begin(), x.size(),
                [&](int i) { x[i] = static_cast<double>(i); });

// Parallelize initialization of `y`
std::fill_n(std::execution::par, y.begin(), y.size(), 2.0);

}

// DAXPY: AX + Y: parallel algorithm version void daxpy(double a, const std::vector &x, std::vector &y) { assert(x.size() == y.size());

std::transform(std::execution::par,
               x.begin(), x.end(),   // input range
               y.begin(),            // second input range
               y.begin(),            // output range
               [&](double xi, double yi) { return a * xi + yi; });

}
*/
Compile and Run

Compiling with support for the parallel algorithms requires:

    g++ and clang++: link against Intel TBB with -ltbb
    nvc++: compile and link with -stdpar flag:
        -stdpar=multicore runs parallel algorithms on CPUs
        -stdpar=gpu runs parallel algorithms on GPUs, further -gpu= flags control the GPU target
        See the Parallel Algorithms Documentation.

The example compiles, runs, and produces correct results as provided. Parallelize it using the C++ standard library parallel algorithms and ensure that the results are still correct. You should see a drastic performance increase when running the program on the GPU (see the solution below if necessary).

The first 3 of the following blocks compile and run the program using different compilers on the CPU.

The last block compiles and runs the program on the GPU. If you get an error, make sure that the lambda captures are capturing scalars by value, and that when capturing a vector to access its data, one captures a pointer to its data by value as well using [x = x.data()].

*/

!g++ -std=c++20 -Ofast -march=native -DNDEBUG -o daxpy exercise3.cpp -ltbb
!./daxpy 1000000

!clang++ -std=c++20 -Ofast -march=native -DNDEBUG -o daxpy exercise3.cpp -ltbb
!./daxpy 1000000

!nvc++ -stdpar=multicore -std=c++20 -O4 -fast -march=native -Mllvm-fast -DNDEBUG -o daxpy exercise3.cpp
!./daxpy 1000000

!nvc++ -stdpar=gpu -std=c++20 -O4 -fast -march=native -Mllvm-fast -DNDEBUG -o daxpy exercise3.cpp
!./daxpy 1000000

