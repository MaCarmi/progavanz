#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define MAX_DIM 2

// Struct Tensore
typedef struct {
    float *data;          /* Buffer dinamico dei dati */
    size_t *shape;        /* Array con le dimensioni del tensore */
    size_t ndim;          /* Numero di dimensioni */
    size_t total_size;    /* Numero totale di elementi */
    int ref_count;        /* Contatore di riferimenti */
} Tensor;

// Enum Tipi gestiti dallo stack
typedef enum {
    VAL_INT,
    VAL_FLOAT,
    VAL_TENSOR,
    VAL_STRING
} ValueType;

// Struct Value (Elemento dello Stack)
typedef struct Value {
    ValueType type;
    union {
        int i_val;
        float f_val;
        Tensor *tensor;
        char *str;
    } as;
    int ref_count;
} Value;

/* Allocazione e deallocazione */
Tensor *tensor_create(const size_t *shape, size_t ndim);
void tensor_retain(Tensor *t);
void tensor_release(Tensor *t);

/* Utility e manipolazione Tensore */
void tensor_fill(Tensor *t, float val);
void tensor_random(Tensor *t);
void tensor_print(const Tensor *t);
int tensor_reshape(Tensor *t, const size_t *new_shape, size_t new_ndim);
void tensor_ravel(Tensor *t);

/* Costruttori e gestione memoria Value */
Value *value_create_int(int val);
Value *value_create_float(float val);
Value *value_create_tensor(Tensor *t);
Value *value_create_string(const char *str);
void value_retain(Value *v);
void value_release(Value *v);
void value_print(const Value *v);

#endif /* TENSOR_H */