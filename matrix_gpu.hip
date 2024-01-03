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

__global__ void myKernel(const float* a, const float* b, const float *biases, float* c, const unsigned int a_cols, const unsigned int blockSize) {
    const unsigned int index = threadIdx.x + (blockIdx.x * blockSize);
    float result = biases[index];
    for (unsigned int i = 0; i < a_cols; ++i) {
        result += a[index * a_cols + i] * b[index];
    }
    c[index] = (2 / (1 + expf(-result))) - 1;
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
    assert(a && b && c);
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == a->cols);
    for (size_t i = 0; i < a->rows * a->cols; ++i) {
        float mutation = ((((float) rand() / (float) RAND_MAX) * 2.0f) - 1.0f) * matrix_t::MUTATION_MAX;
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
    //constexpr unsigned int blockSize = 2;
    //const unsigned int gridSize = c->rows / blockSize;
    const unsigned int gridSize = 1;
    const unsigned int blockSize = c->rows;

    myKernel<<<gridSize, blockSize, 0, hipStreamDefault>>>(a->gpuData, b->gpuData, biases->gpuData, c->gpuData, a->cols, blockSize);
    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());
}

void nn_forward(nn_t *nn) {
    HIP_CHECK(hipMemcpy(nn->layers[0].a->gpuData, nn->layers[0].a->data, nn->layers[0].a->rows * sizeof(float), hipMemcpyHostToDevice));
    for (size_t i = 1; i < nn->num_layers; ++i) {
        matrix_multiply_and_add(nn->layers[i].weights, nn->layers[i - 1].a, nn->layers[i].biases, nn->layers[i].a);
    }
    size_t last_layer = nn->num_layers - 1;
    HIP_CHECK(hipMemcpy(nn->layers[last_layer].a->data, nn->layers[last_layer].a->gpuData, nn->layers[last_layer].a->rows * sizeof(float), hipMemcpyDeviceToHost));
}

#endif
