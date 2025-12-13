# Deeplearning
Portable HPC application using ISO C++
GPU Acceleration using C++ standard library
C++ Prerequistics
Fundamentals of ISO C++ parallism.
Indexing, Ranges and views
Interactive Materials.

C++ Prerequeistis:
ISO C++ Lamda.C++ introduces the Lambda 

std::vector<double> v = {1, 2, 3, 4}
double s = 2;
auto f = [s, &v](int idx){ return v[idx] * s; };
assert(f(1) == 4);
