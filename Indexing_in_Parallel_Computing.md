# Indexing in Parallel Computing and HPC

## Overview

Indexing is a fundamental concept in parallel computing, particularly critical for GPU programming and high-performance computing. Proper indexing ensures efficient memory access, load balancing, and optimal utilization of parallel hardware resources.

## Core Indexing Concepts

### 1. Thread Indexing in CUDA/OpenMP

#### CUDA Thread Hierarchy
```cpp
__global__ void kernel(float* data, int N) {
    // Block and thread indices
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;

    // Global thread index
    int globalIdx = blockId * blockSize + threadId;

    if (globalIdx < N) {
        data[globalIdx] = compute(data[globalIdx]);
    }
}
```

#### Multi-Dimensional Indexing
```cpp
__global__ void matrix_kernel(float* matrix, int width, int height) {
    // 2D thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;  // Row-major ordering
        matrix[idx] = process(matrix[idx]);
    }
}
```

### 2. Memory Layout and Indexing

#### Row-Major vs Column-Major
```cpp
// Row-major (C/C++ style)
#define IDX2D_ROW_MAJOR(i, j, width) ((i) * (width) + (j))

// Column-major (Fortran/MATLAB style)
#define IDX2D_COL_MAJOR(i, j, height) ((j) * (height) + (i))

// 3D indexing
#define IDX3D(i, j, k, width, height) \
    ((i) * (height) * (width) + (j) * (width) + (k))
```

#### Strided Access Patterns
```cpp
// Coalesced access (good for GPU)
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i] = data[i] * 2.0f;  // Contiguous memory access
}

// Strided access (bad for GPU)
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i * stride] = data[i * stride] * 2.0f;  // Non-contiguous
}
```

### 3. Advanced Indexing Techniques

#### Morton/Z-Order Indexing
```cpp
// Morton code (Z-order curve) for spatial locality
uint32_t morton_2d(uint32_t x, uint32_t y) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// Hilbert curve indexing for better locality
uint32_t hilbert_xy_to_index(uint32_t x, uint32_t y, int order) {
    uint32_t index = 0;
    uint32_t s = 1 << (order - 1);

    while (s > 0) {
        uint32_t rx = (x & s) > 0;
        uint32_t ry = (y & s) > 0;

        index += s * s * ((3 * rx) ^ ry);

        // Rotate/flip quadrant
        if (ry == 0) {
            if (rx == 1) {
                x = (1 << order) - 1 - x;
                y = (1 << order) - 1 - y;
            }
            // Swap x and y
            uint32_t temp = x;
            x = y;
            y = temp;
        }
        s >>= 1;
    }
    return index;
}
```

#### Blocked/Indexing for Cache Efficiency
```cpp
// Blocked matrix multiplication indexing
#define BLOCK_SIZE 16

__global__ void blocked_matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < N/BLOCK_SIZE; ++m) {
        // Load blocks into shared memory
        As[ty][tx] = A[row*N + m*BLOCK_SIZE + tx];
        Bs[ty][tx] = B[(m*BLOCK_SIZE + ty)*N + col];

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row*N + col] = sum;
}
```

### 4. Parallel Algorithm Indexing

#### Parallel Prefix Sum (Scan) Indexing
```cpp
// Hillis-Steele parallel scan
__global__ void parallel_scan(float* input, float* output, int n) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2*tid] = input[2*tid];
    temp[2*tid+1] = input[2*tid+1];

    // Build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (tid == 0) temp[n-1] = 0;

    // Traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to output
    output[2*tid] = temp[2*tid];
    output[2*tid+1] = temp[2*tid+1];
}
```

#### Radix Sort Indexing
```cpp
__device__ uint32_t extract_bits(uint32_t value, int bit_start, int num_bits) {
    return (value >> bit_start) & ((1 << num_bits) - 1);
}

__global__ void radix_sort_local(uint32_t* data, uint32_t* temp, int n, int bit_offset) {
    extern __shared__ uint32_t shared_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (gid < n) {
        shared_data[tid] = data[gid];
    }
    __syncthreads();

    // Local histogram
    uint32_t local_histogram[256] = {0};
    for (int i = 0; i < blockDim.x && gid + i * blockDim.x < n; ++i) {
        uint32_t value = shared_data[i];
        uint32_t bits = extract_bits(value, bit_offset, 8);
        local_histogram[bits]++;
    }

    // Prefix sum for local histogram
    uint32_t prefix_sum[256];
    prefix_sum[0] = 0;
    for (int i = 1; i < 256; ++i) {
        prefix_sum[i] = prefix_sum[i-1] + local_histogram[i-1];
    }

    // Scatter to temporary array
    for (int i = 0; i < blockDim.x && gid + i * blockDim.x < n; ++i) {
        uint32_t value = shared_data[i];
        uint32_t bits = extract_bits(value, bit_offset, 8);
        uint32_t pos = prefix_sum[bits]++;
        temp[gid + i * blockDim.x] = value;
    }
}
```

### 5. OpenMP Indexing Patterns

#### Loop Indexing with OpenMP
```cpp
void openmp_indexing(float* data, int N) {
    // Static scheduling
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        data[i] = process(data[i]);
    }

    // Dynamic scheduling for load balancing
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        data[i] = complex_process(data[i]);
    }

    // Guided scheduling
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; ++i) {
        data[i] = variable_workload(data[i]);
    }
}
```

#### Multi-dimensional OpenMP Indexing
```cpp
void matrix_openmp(float** matrix, int rows, int cols) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = compute(matrix[i][j]);
        }
    }
}
```

### 6. C++ Parallel Algorithm Indexing

#### Parallel Transform with Custom Indexing
```cpp
#include <execution>
#include <algorithm>
#include <vector>

void parallel_transform_indexing(std::vector<float>& data) {
    // Parallel transform with index awareness
    std::transform(std::execution::par, data.begin(), data.end(), data.begin(),
                   [idx = 0](float val) mutable {
                       // Access neighboring elements
                       float result = val;
                       if (idx > 0) result += data[idx - 1] * 0.1f;
                       if (idx < data.size() - 1) result += data[idx + 1] * 0.1f;
                       ++idx;
                       return result;
                   });
}
```

#### Indexed Parallel Reduction
```cpp
#include <numeric>

float indexed_parallel_reduce(const std::vector<float>& data) {
    // Parallel reduction with index-based weighting
    return std::transform_reduce(
        std::execution::par,
        data.begin(), data.end(),
        0.0f,
        std::plus<float>{},
        [idx = 0](float val) mutable {
            float weight = 1.0f / (idx + 1);  // Example weighting
            ++idx;
            return val * weight;
        }
    );
}
```

### 7. Advanced Indexing Strategies

#### Space-Filling Curves
```cpp
// Peano curve for 2D indexing
std::vector<std::pair<int, int>> peano_curve(int order) {
    std::vector<std::pair<int, int>> curve;
    // Implementation of Peano space-filling curve
    // Used for cache-efficient traversal of 2D data
    return curve;
}

// Gray code indexing for error detection
uint32_t gray_encode(uint32_t n) {
    return n ^ (n >> 1);
}

uint32_t gray_decode(uint32_t n) {
    uint32_t mask = n;
    while (mask) {
        mask >>= 1;
        n ^= mask;
    }
    return n;
}
```

#### Hash-Based Indexing
```cpp
// Perfect hash functions for GPU
__device__ uint32_t perfect_hash(uint32_t key, uint32_t table_size) {
    // Minimal perfect hash implementation
    // Ensures no collisions for known key set
    return key % table_size;  // Simplified example
}

// Cuckoo hashing for GPU
__device__ uint32_t cuckoo_hash(uint32_t key, int function_id) {
    const uint32_t hash_functions[4] = {
        0x9E3779B1, 0xB5297A4D, 0x68E31DA4, 0x1B56C4E9
    };
    return (key ^ hash_functions[function_id]) % TABLE_SIZE;
}
```

### 8. Performance Optimization

#### Memory Coalescing
```cpp
// Good: Coalesced memory access
__global__ void coalesced_access(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Bad: Non-coalesced memory access
__global__ void non_coalesced_access(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx * stride] = data[idx * stride] * 2.0f;
    }
}
```

#### Bank Conflict Avoidance
```cpp
// Shared memory bank conflict free access
__shared__ float shared_data[1024];

__global__ void bank_conflict_free(float* data) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Good: No bank conflicts
    shared_data[tid] = data[bid * blockDim.x + tid];
    shared_data[tid + 32] = data[bid * blockDim.x + tid + 32];

    __syncthreads();

    // Process data
    data[bid * blockDim.x + tid] = shared_data[tid] + shared_data[tid + 32];
}
```

### 9. Debugging Indexing Issues

#### Bounds Checking
```cpp
#define DEBUG_INDEXING 1

__device__ void check_bounds(int idx, int max_idx, const char* location) {
    #ifdef DEBUG_INDEXING
    if (idx < 0 || idx >= max_idx) {
        printf("Index out of bounds at %s: %d (max: %d)\n", location, idx, max_idx);
        __trap();
    }
    #endif
}

__global__ void safe_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    check_bounds(idx, N, "kernel entry");

    if (idx < N) {
        data[idx] = compute(data[idx]);
    }
}
```

#### Race Condition Detection
```cpp
// Atomic operations for safe indexing
__global__ void atomic_indexing(int* counters, int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Atomic increment for unique index assignment
    int unique_idx = atomicAdd(&counters[0], 1);

    if (unique_idx < MAX_DATA) {
        data[unique_idx] = process(idx);
    }
}
```

## Conclusion

Proper indexing is crucial for achieving optimal performance in parallel computing and HPC applications. Key principles include:

1. **Memory Coalescing**: Ensure contiguous memory access patterns
2. **Load Balancing**: Distribute work evenly across processing units
3. **Cache Efficiency**: Use space-filling curves and blocking techniques
4. **Bank Conflict Avoidance**: Optimize shared memory access patterns
5. **Bounds Checking**: Implement safety checks in debug builds
6. **Algorithm Selection**: Choose appropriate parallel algorithms for data access patterns

Effective indexing can improve performance by orders of magnitude in parallel computing environments, making it one of the most important considerations in HPC programming.</content>
<parameter name="filePath">/home/rahul/AiStätt/cpp/Indexing_in_Parallel_Computing.md