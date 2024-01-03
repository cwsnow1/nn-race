#include <iostream>

#include "../matrix.h"
#include <hip/hip_runtime.h>

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

int main() {
    matrix_t *a = matrix_make(20, 30);
    matrix_t *b = matrix_make(30, 1);
    matrix_t *biases = matrix_make(20, 1);
    matrix_randomize(a);
    matrix_randomize(b);
    matrix_randomize(biases);

    HIP_CHECK(hipMemcpy(a->gpuData, a->data, a->rows * a->cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b->gpuData, b->data, b->rows * b->cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(biases->gpuData, biases->data, biases->rows * biases->cols * sizeof(float), hipMemcpyHostToDevice));

    matrix_t *c = matrix_make(20, 1);

    matrix_multiply_and_add(a, b, biases, c);

    HIP_CHECK(hipMemcpy(c->data, c->gpuData, c->rows * c->cols * sizeof(float), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < 20; ++i) {
        std::cout << c->data[i] << "\n";
    }


    return 0;
}