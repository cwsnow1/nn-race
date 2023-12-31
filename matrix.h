#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct matrix_s {
    float *data;
    size_t rows;
    size_t cols;
} matrix_t;


matrix_t *matrix_make(int rows, int cols);
void matrix_multiply(matrix_t *a, matrix_t *b, matrix_t *c);
void matrix_add(matrix_t *a, matrix_t *b, matrix_t *c);
void matrix_sigmoid(matrix_t *a);
void matrix_randomize(matrix_t *m);
void matrix_average(matrix_t *a, matrix_t *b, matrix_t *c);
void matrix_delete(matrix_t *m);

#define MATRIX_AT(mat, row, col)    ((mat)->data[(mat)->cols * (row) + (col)])

#endif