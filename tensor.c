/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

Tensor *tensor_create(const size_t *shape, size_t ndim) {
    if (ndim == 0 || ndim > MAX_DIM) return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!t->shape) {
        free(t);
        return NULL;
    }
    
    t->total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->total_size *= shape[i];
    }

    t->data = (float *)malloc(t->total_size * sizeof(float));
    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }

    t->ref_count = 1;
    t->mmap_base = NULL;
    t->mmap_size = 0;
    t->owns_data = 1;

    return t;
}

void tensor_retain(Tensor *t) {
    if (t) t->ref_count++;
}

void tensor_release(Tensor *t) {
    if (!t) return;
    t->ref_count--;
    if (t->ref_count <= 0) {
        if (t->owns_data && t->data) {
            free(t->data);
        }
        free(t->shape);
        free(t);
    }
}

void tensor_fill(Tensor *t, float val) {
    if (!t) return;
    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] = val;
    }
}

void tensor_random(Tensor *t) {
    if (!t) return;
    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }
}

/* Stampa conforme alle specifiche: Tensor(shape=[...], data=[...]) */
void tensor_print(const Tensor *t) {
    if (!t) return;
    printf("Tensor(shape=[");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu%s", t->shape[i], (i < t->ndim - 1) ? " " : "");
    }
    printf("], data=[");
    for (size_t i = 0; i < t->total_size; i++) {
        printf("%.2f%s", t->data[i], (i < t->total_size - 1) ? " " : "");
    }
    printf("])\n");
}

int tensor_reshape(Tensor *t, const size_t *new_shape, size_t new_ndim) {
    if (!t || new_ndim == 0 || new_ndim > MAX_DIM) return 0;
    
    size_t new_total = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_total *= new_shape[i];
    }

    if (new_total != t->total_size) return 0;

    size_t *tmp_shape = (size_t *)realloc(t->shape, new_ndim * sizeof(size_t));
    if (!tmp_shape) return 0;

    t->shape = tmp_shape;
    for (size_t i = 0; i < new_ndim; i++) {
        t->shape[i] = new_shape[i];
    }
    t->ndim = new_ndim;

    return 1;
}

void tensor_ravel(Tensor *t) {
    if (!t) return;
    size_t *new_shape = (size_t *)realloc(t->shape, sizeof(size_t));
    if (!new_shape) return;

    t->shape = new_shape;
    t->shape[0] = t->total_size;
    t->ndim = 1;
}

/* Operatore #: Genera un tensore 1D con le dimensioni di t */
Tensor *tensor_get_shape(const Tensor *t) {
    if (!t) return NULL;
    size_t shape_dim = t->ndim;
    Tensor *s = tensor_create(&shape_dim, 1);
    if (!s) return NULL;
    for (size_t i = 0; i < t->ndim; i++) {
        s->data[i] = (float)t->shape[i];
    }
    return s;
}

/* --- IMPLEMENTAZIONE VALUE --- */

Value *value_create_int(int val) {
    Value *v = (Value *)malloc(sizeof(Value));
    if (!v) return NULL;
    v->type = VAL_INT;
    v->as.i_val = val;
    v->ref_count = 1;
    return v;
}

Value *value_create_float(float val) {
    Value *v = (Value *)malloc(sizeof(Value));
    if (!v) return NULL;
    v->type = VAL_FLOAT;
    v->as.f_val = val;
    v->ref_count = 1;
    return v;
}

Value *value_create_tensor(Tensor *t) {
    if (!t) return NULL;
    Value *v = (Value *)malloc(sizeof(Value));
    if (!v) return NULL;
    v->type = VAL_TENSOR;
    v->as.tensor = t;
    tensor_retain(t);
    v->ref_count = 1;
    return v;
}

Value *value_create_string(const char *str) {
    if (!str) return NULL;
    Value *v = (Value *)malloc(sizeof(Value));
    if (!v) return NULL;
    v->type = VAL_STRING;
    v->as.str = strdup(str);
    v->ref_count = 1;
    return v;
}

void value_retain(Value *v) {
    if (v) v->ref_count++;
}

void value_release(Value *v) {
    if (!v) return;
    v->ref_count--;
    if (v->ref_count <= 0) {
        if (v->type == VAL_TENSOR) {
            tensor_release(v->as.tensor);
        } else if (v->type == VAL_STRING) {
            free(v->as.str);
        }
        free(v);
    }
}

void value_print(const Value *v) {
    if (!v) return;
    switch (v->type) {
        case VAL_INT: printf("%d", v->as.i_val); break;
        case VAL_FLOAT: printf("%f", v->as.f_val); break;
        case VAL_STRING: printf("\"%s\"", v->as.str); break;
        case VAL_TENSOR: tensor_print(v->as.tensor); break;
    }
}x