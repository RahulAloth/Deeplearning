## Introducing DAXPY

To begin our code-based exploration of C++ parallel algorithms and their related features, we will work with Double-precision AX Plus Y (DAXPY)
, which is one of the main functions in the standard Basic Linear Algebra Subroutines (BLAS) library.

### Learning objective
By the time you complete this notebook you should:

    Know what the DAXPY operation does
    Be familiar with the sequential C++ starting point DAXPY application
    Be able to compile and run the DAXPY application
    Have a performance baseline against which to compare the improvements you will be making in later notebooks


## DAXPY

The DAXPY operation is a combination of scalar multiplication and vector addition. It takes two vectors of 64-bit floats, x and y and a scalar value a. It then multiplies each element x[i] by a and adds the result to y[i].


### A Sequential Starting Point

Throughout the course of the next several notebooks, you will work to refactor a sequential DAXPY application to run on the GPU.

A working implementation is provided in starting_point.cpp. Please take several minutes to read through the application before proceeding to the next part of this notebook. We assume that your level of C++ programming is such that the entirety of the application will make sense to you. If you find yourself unfamiliar with any parts of the application, please spend any time you need to bring yourself up to speed.

## The "Core" of the Implementation

The "core" of the sequential implementation provided in starting_point.cpp is split into two separate functions, initialize and daxpy:

/// Intialize vectors `x` and `y`: raw loop sequential version
void initialize(std::vector<double> &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = (double)i;
    y[i] = 2.;
  }
}

/// DAXPY: AX + Y: raw loop sequential version
void daxpy(double a, std::vector<double> const &x, std::vector<double> &y) {
  assert(x.size() == y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] += a * x[i];
  }
}


You will notice that we initialize the vectors such that `x[i] = i` and `y[i] = 2.` We are doing this, rather than using some form of random initialization, in order that we can reliably and easily verify the correctness of the `daxpy` function.

The `daxpy` function itself implements a loop over all vector elements, reading from both `x` and `y` and writing the solution to `y`.

!g++ -std=c++11 -Ofast -march=native -DNDEBUG -o daxpy starting_point.cpp
!./daxpy 10000000

From Raw DAXPY Loop to Serial C++ Transform Algorithm

In this notebook you will perform your first exercise, refactoring the raw loop in daxpy to instead use the C++ standard library algorithm transform.
Learning Objectives

By the time you complete this notebook you should:

    Understand the importance of functions in the C++ algorithms library to your ability to write "parallel first" applications
    Be familiar with the transform algorithm
    Be able to refactor the raw loop in daxpy to use transform instead

The Need to Refactor Raw Loops

As discussed in the presentation earlier in the course, we can leverage parallelism in standard C++ code by way of the execution policies available in functions of the C++ algorithms library.

To that end, in order to make our applications capable to run in parallel, and in particular on massively parallel GPUs, we must look for opportunities to leverage the functions in the C++ algorithms library that provide an execution policy. For-loops are very often readily available to be refactored in just this way.

We will not actually implement parallelism into the DAXPY application until a later notebook, however, in this notebook you will make the first prerequisite refactor to the daxpy function in order that it can be made parallel at a later stage by replacing its raw loop with the C++ algorithm transform.

If you are unfamiliar with the use of standard library algorithms such as transform, or with any of their associated constructs, such as iterators, lambdas, unary operators, binary operators etc., it will be worth your time to come up to speed on them. Aside from being succinct and performant, your ability to write applications that are "parallel first" and capable of benifiting from GPU parallelism will hinge largely on your ability to work fluently with standard library algorithms.
The transform Algorithm

Please take a moment to review the transform algorithm. For the following exercise you will be using overload number (3).
Exercise 1: From Raw DAXPY Loop to Serial C++ Transform Algorithm

For this exercise you will work with exercise1.cpp, which is largely identical to the starting_point.cpp file you are already familiar with except that it includes two TODO comments that indicate where you need to add or refactor code.

To complete this first exercise, the daxpy function needs to be rewritten to use the C++ standard library algorithm transform and this will require adding some headers. Below are the parts of the file containing TODOs.

