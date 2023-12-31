#include <hip/hip_runtime.h>
#include <iostream>

#include "matrix.h"
#include "nn.h"

#ifndef MATRIX_CPU

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

/// \brief Multiplies matrices \p A and \p B and stores the result to \p C.
/// - The number of rows of the result matrix is equal to the number of rows of matrix A
///   which is \p blockDim.y*gridDim.y.
/// - The number of columns of the result matrix is equal to the number of columns of matrix B
///   which is \p blockDim.x*gridDim.x.
/// - The number of columns of matrix \p A is passed as argument.
/// - The matrix elements are stored in a row-major order.
///
/// - Each thread in the grid is responsible for one element of the result matrix.
/// - Each element is calculated cooperatively in a tiled manner. In each step, a BlockSize*BlockSize
///   tile is loaded to the shared memory so individual threads can address this shared cache instead
///   of loading the same values from the global device memory individually.
///   The end result is accumulated through each step on a per-thread basis.
/// - The matrix dimensions are assumed to be multiples of the block size for simplicity.
template<unsigned int BlockSize>
__global__ void matrix_multiplication_kernel(const float*       A,
                                             const float*       B,
                                             const float*       biases,
                                             float*             C,
                                             const unsigned int a_cols)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    // b_cols must match the number of output matrix columns.
    const unsigned int b_cols = blockDim.x * gridDim.x;

    // The number of tiles is determined by A's columns (which is equal to B's rows).
    const unsigned int steps = a_cols / BlockSize;

    // thread_result is the accumulation variable.
    float thread_result = 0.0F;
    for(unsigned int step = 0; step < steps; step++)
    {
        // Shared memory is used to cache the tile from both input matrices.
        // The tile is a square of BlockSize*BlockSize.
        __shared__ float a_values[BlockSize][BlockSize];
        __shared__ float b_values[BlockSize][BlockSize];

        // Index of the top-left element of the tile in A.
        // "BlockSize * a_cols * by" is the number of elements to move "down".
        // "BlockSize * step" is the number of elements to move "right".
        const unsigned int a_idx = BlockSize * (a_cols * by + step);

        // Index of the top-left element of the tile in B.
        // "BlockSize * b_cols * step" is the number of elements to move "down".
        // "BlockSize * bx" is the number of elements to move "right".
        const unsigned int b_idx = BlockSize * (b_cols * step + bx);

        // Load each element in the tile to shared memory.
        a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
        b_values[ty][tx] = B[b_idx + b_cols * ty + tx];

        // Synchronization is needed to make sure that all elements are loaded before
        // starting the calculation.
        __syncthreads();

        // Each thread calculates the scalar product of the tile and increments the
        // thread-individual thread_result.
        for(unsigned int i = 0; i < BlockSize; i++)
        {
            thread_result += a_values[ty][i] * b_values[i][tx];
        }

        // Synchronize to ensure that the calculation is finished before the next tile's
        // elements start to load.
        __syncthreads();
    }

    // Calculate the index of the top-left element of the output block.
    const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

    // Every thread stores the final result to global memory.
    float res = thread_result + biases[block_offset + b_cols * ty + tx];
    C[block_offset + b_cols * ty + tx] = (2 / (1 + expf(-res))) - 1;
}

matrix_t *matrix_make(int rows, int cols) {
    matrix_t *m = (matrix_t*) malloc(sizeof(matrix_t));
    m->cols = cols;
    m->rows = rows;
    m->data = (float*) malloc(sizeof(float) * rows * cols);
    HIP_CHECK(hipMalloc(&m->gpuData, sizeof(float) * rows * cols));
    return m;
}

void matrix_delete(matrix_t *m) {
    if (m == NULL) return;
    HIP_CHECK(hipFree(m->gpuData));
    free(m->data);
    free(m);
}

void matrix_average(matrix_t *a, matrix_t *b, matrix_t *c) {
    constexpr float MUTATION_MAX = 0.01f;
    assert(a && b && c);
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == a->cols);
    for (size_t i = 0; i < a->rows * a->cols; ++i) {
        float mutation = ((((float) rand() / (float) RAND_MAX) * 2.0f) - 1.0f) * MUTATION_MAX;
        c->data[i] = (a->data[i] + b->data[i]) / 2.0f + mutation;
    }
    HIP_CHECK(hipMemcpy(c->gpuData, c->data, c->rows * c->cols * sizeof(float), hipMemcpyHostToDevice));
}

void matrix_randomize(matrix_t *m) {
    for (size_t i = 0; i < m->rows * m->cols; ++i) {
        m->data[i] = (((float) rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
    }
    HIP_CHECK(hipMemcpy(m->gpuData, m->data, m->rows * m->cols * sizeof(float), hipMemcpyHostToDevice));
}

void matrix_multiply_and_add(matrix_t *a, matrix_t *b, matrix_t *biases, matrix_t *c) {
    constexpr unsigned int block_size = 1;

    const dim3 block_dim(block_size, block_size);
    const dim3 grid_dim(c->cols / block_size, c->rows / block_size);

    matrix_multiplication_kernel<block_size>
        <<<grid_dim, block_dim, 0, hipStreamDefault>>>(a->gpuData, b->gpuData, biases->gpuData, c->gpuData, a->cols);
    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());
}

void nn_forward(nn_t *nn) {
    HIP_CHECK(hipMemcpy(nn->layers[0].a->gpuData, nn->layers[0].a->data, nn->layers[0].a->rows * sizeof(float), hipMemcpyHostToDevice));
    for (size_t i = 1; i < nn->num_layers; ++i) {
        matrix_multiply_and_add(nn->layers[i].weights, nn->layers[i - 1].a, nn->layers[i].biases, nn->layers[i].a);
        //matrix_add(nn->layers[i].a, nn->layers[i].biases, nn->layers[i].a);
        //matrix_sigmoid(nn->layers[i].a);
    }
    size_t last_layer = nn->num_layers - 1;
    HIP_CHECK(hipMemcpy(nn->layers[last_layer].a->data, nn->layers[last_layer].a->gpuData, nn->layers[last_layer].a->rows * sizeof(float), hipMemcpyDeviceToHost));
}

#endif
