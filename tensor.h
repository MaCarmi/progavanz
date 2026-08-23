/*
Marco Carmignano
SM3201526
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define MAX_DIM 2

typedef struct {
    float *data;          /* Buffer dinamico dei dati */
    size_t *shape;        /* Array con le dimensioni del tensore */
    size_t ndim;          /* Numero di dimensioni */
    size_t total_size;    /* Numero totale di elementi */
    int ref_count;        /* Contatore di riferimenti */

    void *mmap_base;
    size_t mmap_size;
    int owns_data;
} Tensor;


/* Allocazione e deallocazione */

Tensor *tensor_create(const size_t *shape, size_t ndim);

void tensor_retain(Tensor *t);

void tensor_release(Tensor *t);


/* Utility e manipolazione */

void tensor_fill(Tensor *t, float val);

void tensor_random(Tensor *t);

void tensor_print(const Tensor *t);

int tensor_reshape(
    Tensor *t,
    const size_t *new_shape,
    size_t new_ndim
);

void tensor_ravel(Tensor *t);

#endif /* TENSOR_H */