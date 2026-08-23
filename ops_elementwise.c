/*
Marco Carmignano
SM3201526
*/

#include "ops_elementwise.h"

#include <stddef.h>

/*
Verifica se due tensori hanno la stessa forma

Riceve in input: a, b che sarebbero i tensori da confrontare

Restituisce in output: 1 se hanno la stessa forma, 0 altrimenti 
*/
static int same_shape(const Tensor *a, const Tensor *b){
    if (a == NULL || b == NULL) {
        return 0;
    }
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

/*
Crea un nuovo tensore con la stessa forma di un tensore dato

Prende in input: a che è il tensore di riferimento

Restituisce in output: un puntatore al nuovo tensore creato, o NULL in caso di errore
*/
static Tensor *create_like(const Tensor *a) {
    if (a == NULL) {
        return NULL;
    }
    return tensor_create(a->shape, a->ndim);
}

/*
Somma due tensori elemento per elemento

Prende in input: a, b che sono i tensori da sommare 
Restituisce in output: out che è il puntatore al tensore risultato della sommma, che contiene a + b

In particolare ritorna:
TF_OK se l'operazione è andata a buon fine
TF_ERR_NULL_ARGUMENT se uno dei puntatori passati è NULL
TF_ERR_DIMENSION_MISMATCH se i tensori hanno forme diverse
TF_ERR_ALLOCATION se si verifica un errore di allocazione
*/
TFStatus tf_add(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    *out = result;

    return TF_OK;
}

/*
Sottrae due tesnori elemento per elemento

Prende in input: a,b che sono i tensori con la stessa forma

Restituisce in output: out che è il puntatore al tensore risultato della sottrazione, che contiene a - b
*/

TFStatus tf_sub(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    *out = result;

    return TF_OK;
}

/*
Restituisce il prodotto di due tensori elemento per elemento

Riceve in input: a, b che sono i tensori da moltiplicare
Restituisce in output: out che è il puntatore al tensore risultato della moltiplicazione, che contiene a * b
*/

TFStatus tf_mul(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    *out = result;

    return TF_OK;
}

/*
Confronta due tensori pelemento per elemento (equivale a <)

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] < b[i] o 0.0f altrimenti
*/

TFStatus tf_less(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] < b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Confronta due tensori elemento per elemento (equivale a >)

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] > b[i] o 0.0f altrimenti
*/

TFStatus tf_greater(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Confronta due tensori elemento per elemento (equivale a ==)

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] == b[i] o 0.0f altrimenti
*/

TFStatus tf_equal(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] == b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Confronta due tensori elemento per elemento (equivale a &&)

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] && b[i] o 0.0f altrimenti  
*/

TFStatus tf_and(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] =
            (a->data[i] != 0.0f && b->data[i] != 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Confronta due tensori elemento per elemento (equivale a ||)

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] || b[i] o 0.0f altrimenti  
*/

TFStatus tf_or(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] =
            (a->data[i] != 0.0f || b->data[i] != 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Confronta un tensore elemento per elemento (equivale a !)

Restituisce in input: a che è il tensore da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene 1.0f se a[i] == 0.0f o 0.0f altrimenti
*/
TFStatus tf_not(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] == 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Seleziona tra due tensori in base a una maschera

Restituisce in input: a, b che sono i tensori da selezionare e mask che è la maschera di selezione
Restituisce in output: out che è il puntatore al tensore risultato della selezione, che contiene a[i] se mask[i] == 1.0f o b[i] se mask[i] == 0.0f
*/

TFStatus tf_select(
    const Tensor *b,
    const Tensor *a,
    const Tensor *mask,
    Tensor **out)
{
    if (a == NULL || b == NULL || mask == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b) || !same_shape(a, mask)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (mask->data[i] == 1.0f)
            ? a->data[i]
            : b->data[i];
    }

    *out = result;

    return TF_OK;
}


/*
ReLU elemento per elemento 

Restituisce in input: a che è il tensore da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della ReLU, che contiene max(0, a[i])
*/
TFStatus tf_relu(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;
    }

    *out = result;

    return TF_OK;
}

/*
Restituisce il minimo tra due tensori elemento per elemento

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene min(a[i], b[i])
*/
TFStatus tf_min(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] < b->data[i])
            ? a->data[i]
            : b->data[i];
    }

    *out = result;

    return TF_OK;
}

/*
Restituisce il massimo tra due tensori elemento per elemento

Restituisce in input: a, b che sono i tensori da confrontare
Restituisce in output: out che è il puntatore al tensore risultato della comparazione, che contiene max(a[i], b[i])
*/
TFStatus tf_max(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (!same_shape(a, b)) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    Tensor *result = create_like(a);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > b->data[i])
            ? a->data[i]
            : b->data[i];
    }

    *out = result;

    return TF_OK;
}

/*
Somma tutti gli elementi di un tensore

Restituisce in input: a che è il tensore da sommare
Restituisce in output: out che è il puntatore al tensore risultato della somma, che contiene la somma di tutti gli elementi di a
*/
TFStatus tf_sum(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    size_t shape[1] = {1};

    Tensor *result = tensor_create(shape, 1);

    if (result == NULL) {
        return TF_ERR_ALLOCATION;
    }

    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a->total_size; i++) {
        sum += a->data[i];
    }

    result->data[0] = sum;

    *out = result;

    return TF_OK;
}