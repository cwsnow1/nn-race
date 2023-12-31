#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>

#include "matrix.h"

typedef struct layer_s {
    matrix_t *weights;
    matrix_t *biases;
    matrix_t *a;
} layer_t;

typedef struct nn_s {
    layer_t *layers;
    size_t num_layers;
} nn_t;

nn_t *nn_init(size_t num_layers, size_t *layer_sizes);
nn_t *nn_average(nn_t *nn_a, nn_t *nn_b);
void nn_delete(nn_t *me);

#endif
