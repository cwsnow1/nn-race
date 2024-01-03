#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#include "nn.h"


layer_t make_layer(size_t layer_size, size_t prev_layer_size, bool randomize) {
    layer_t me;
    me.a = matrix_make(layer_size, 1);
    me.weights = matrix_make(layer_size, prev_layer_size);
    me.biases = matrix_make(layer_size, 1);

    if (randomize) {
        matrix_randomize(me.weights);
        matrix_randomize(me.biases);
    }
    return me;
}

layer_t make_input_layer(size_t layer_size) {
    layer_t me;
    me.a = matrix_make(layer_size, 1);
    me.weights = NULL;
    me.biases = NULL;
    return me;
}

nn_t* nn_average(nn_t* nn_a, nn_t* nn_b) {
    assert(nn_a->num_layers == nn_b->num_layers);
    nn_t* avg = (nn_t*)malloc(sizeof(nn_t));
    avg->num_layers = nn_a->num_layers;
    avg->layers = (layer_t*)malloc(sizeof(layer_t) * avg->num_layers);
    avg->layers[0] = make_input_layer(nn_a->layers[0].a->rows);
    for (size_t i = 1; i < avg->num_layers; ++i) {
        avg->layers[i] = make_layer(nn_a->layers[i].a->rows, nn_a->layers[i - 1].a->rows, false);
        matrix_average(nn_a->layers[i].a, nn_b->layers[i].a, avg->layers[i].a);
        matrix_average(nn_a->layers[i].weights, nn_b->layers[i].weights, avg->layers[i].weights);
        matrix_average(nn_a->layers[i].biases, nn_b->layers[i].biases, avg->layers[i].biases);
    }
    return avg;
}

nn_t* nn_init(size_t num_layers, size_t* layer_sizes) {
    nn_t* me = (nn_t*)malloc(sizeof(nn_t));
    me->num_layers = num_layers;
    me->layers = (layer_t*)malloc(sizeof(layer_t) * num_layers);
    me->layers[0] = make_input_layer(layer_sizes[0]);
    for (size_t i = 1; i < num_layers; ++i) {
        me->layers[i] = make_layer(layer_sizes[i], layer_sizes[i - 1], true);
    }
    return me;
}

void nn_delete(nn_t* me) {
    for (size_t i = 0; i < me->num_layers; ++i) {
        matrix_delete(me->layers[i].a);
        matrix_delete(me->layers[i].weights);
        matrix_delete(me->layers[i].biases);
    }
    free(me);
}
