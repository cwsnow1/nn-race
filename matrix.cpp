#include <stdlib.h>
//#define NDEBUG
#include <assert.h>
#include <math.h>

#include "matrix.h"
#include "nn.h"
#ifdef MATRIX_CPU

matrix_t* matrix_make(int rows, int cols) {
    matrix_t* m = (matrix_t*)malloc(sizeof(matrix_t));
    m->cols = cols;
    m->rows = rows;
    m->data = (float*)malloc(sizeof(float) * rows * cols);
    return m;
}

void matrix_multiply(matrix_t* a, matrix_t* b, matrix_t* c) {
    assert(a && b && c);
    assert(a->cols == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == b->cols);
    for (size_t i = 0; i < c->rows; ++i) {
        float result = 0.0f;
        for (size_t k = 0; k < b->rows; ++k) {
            result += MATRIX_AT(a, i, k) * b->data[k];
        }
        c->data[i] = result;
    }
}

void matrix_add(matrix_t* a, matrix_t* b, matrix_t* c) {
    assert(a && b && c);
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == a->cols);
    for (size_t i = 0; i < a->rows * a->cols; ++i) {
        c->data[i] = a->data[i] + b->data[i];
    }
}

float sigmoidf(float x) {
    return (2 / (1 + expf(-x))) - 1;
}

void matrix_sigmoid(matrix_t* a) {
    for (size_t i = 0; i < a->rows * a->cols; ++i) {
        a->data[i] = sigmoidf(a->data[i]);
    }
}

void matrix_randomize(matrix_t* m) {
    for (size_t i = 0; i < m->rows * m->cols; ++i) {
        m->data[i] = (((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
    }
}

void matrix_average(matrix_t* a, matrix_t* b, matrix_t* c) {
    assert(a && b && c);
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);
    assert(c->rows == a->rows);
    assert(c->cols == a->cols);
    for (size_t i = 0; i < a->rows * a->cols; ++i) {
        float mutation = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f) * matrix_t::MUTATION_MAX;
        c->data[i] = (a->data[i] + b->data[i]) / 2.0f + mutation;
    }
}

void matrix_delete(matrix_t* m) {
    if (m == NULL) return;
    free(m->data);
    free(m);
}

void nn_forward(struct nn_s* nn) {
    for (size_t i = 1; i < nn->num_layers; ++i) {
        matrix_multiply(nn->layers[i].weights, nn->layers[i - 1].a, nn->layers[i].a);
        matrix_add(nn->layers[i].a, nn->layers[i].biases, nn->layers[i].a);
        matrix_sigmoid(nn->layers[i].a);
    }
}
#endif
