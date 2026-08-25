/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#define _POSIX_C_SOURCE 200809L

#include "io_tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#define TENSOR_DATA_OFFSET 64


typedef struct {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
} OnDiskTensor;


/*
Calcola il numero totale di elementi in un tensore dato il suo shape e il numero di dimensioni.

Prende in input: shape che è un array con le dimensioni del tensore e ndim che è il numero di dimensioni
Restituisce in output: il numero totale di elementi nel tensore
 */
static size_t compute_total_size(const size_t *shape, size_t ndim)
{
    size_t total = 1;

    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }

    return total;
}


/*
Salva un tensore in un file nel formato TensorForth.

Prende in input: a che è il tensore da salvare e filename che è il percorso del file da creare
Restituisce in output: TF_OK se l'operazione è andata a buon fine, altrimenti un codice di errore
 */
TFStatus tf_write_tensor_file(const Tensor *a, const char *filename)
{
    if (a == NULL || filename == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (a->ndim == 0 || a->ndim > MAX_DIM) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        return TF_ERR_FILE;
    }

    OnDiskTensor header;

    for (size_t i = 0; i < MAX_DIM; i++) {
        header.shape[i] = 0;
    }

    for (size_t i = 0; i < a->ndim; i++) {
        header.shape[i] = (int32_t)a->shape[i];
    }

    header.ndim = (int32_t)a->ndim;
    header.data_offset = TENSOR_DATA_OFFSET;

    if (fwrite(&header, sizeof(OnDiskTensor), 1, fp) != 1) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    /*
     * Padding fino all'offset 64.
     */
    size_t written_header = sizeof(OnDiskTensor);

    if (written_header > TENSOR_DATA_OFFSET) {
        fclose(fp);
        return TF_ERR_INVALID_FORMAT;
    }

    unsigned char zero = 0;

    for (size_t i = written_header; i < TENSOR_DATA_OFFSET; i++) {
        if (fwrite(&zero, sizeof(unsigned char), 1, fp) != 1) {
            fclose(fp);
            return TF_ERR_FILE;
        }
    }

    if (fwrite(a->data, sizeof(float), a->total_size, fp) != a->total_size) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    if (fclose(fp) != 0) {
        return TF_ERR_FILE;
    }

    return TF_OK;
}


/*
Legge un tensore da un file usando mmap.

Prende in input: filename che è il percorso del file da leggere
Restituisce in output: out che è il puntatore al tensore risultato della lettura, che contiene i valori del tensore letto
 */
TFStatus tf_read_tensor_mmap(const char *filename, Tensor **out)
{
    if (filename == NULL || out == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    FILE *fp = fopen(filename, "rb");

    if (fp == NULL) {
        return TF_ERR_FILE;
    }

    int fd = fileno(fp);

    if (fd == -1) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    struct stat st;

    if (fstat(fd, &st) != 0) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    if (st.st_size < (off_t)TENSOR_DATA_OFFSET) {
        fclose(fp);
        return TF_ERR_INVALID_FORMAT;
    }

    void *mapped = mmap(
        NULL,
        (size_t)st.st_size,
        PROT_READ,
        MAP_SHARED,
        fd,
        0
    );

    if (mapped == MAP_FAILED) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    /*
     * Dopo mmap il file può essere chiuso:
     * il mapping resta valido fino a munmap.
     */
    fclose(fp);

    OnDiskTensor *header = (OnDiskTensor *)mapped;

    if (header->ndim <= 0 || header->ndim > MAX_DIM) {
        munmap(mapped, (size_t)st.st_size);
        return TF_ERR_INVALID_FORMAT;
    }

    if (header->data_offset != TENSOR_DATA_OFFSET) {
        munmap(mapped, (size_t)st.st_size);
        return TF_ERR_INVALID_FORMAT;
    }

    size_t ndim = (size_t)header->ndim;

    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));

    if (shape == NULL) {
        munmap(mapped, (size_t)st.st_size);
        return TF_ERR_ALLOCATION;
    }

    for (size_t i = 0; i < ndim; i++) {
        if (header->shape[i] <= 0) {
            free(shape);
            munmap(mapped, (size_t)st.st_size);
            return TF_ERR_INVALID_FORMAT;
        }

        shape[i] = (size_t)header->shape[i];
    }

    size_t total_size = compute_total_size(shape, ndim);

    size_t required_size =
        TENSOR_DATA_OFFSET + total_size * sizeof(float);

    if ((size_t)st.st_size < required_size) {
        free(shape);
        munmap(mapped, (size_t)st.st_size);
        return TF_ERR_INVALID_FORMAT;
    }

    Tensor *result = (Tensor *)malloc(sizeof(Tensor));

    if (result == NULL) {
        free(shape);
        munmap(mapped, (size_t)st.st_size);
        return TF_ERR_ALLOCATION;
    }

    result->shape = shape;
    result->ndim = ndim;
    result->total_size = total_size;
    result->ref_count = 1;

    result->mmap_base = mapped;
    result->mmap_size = (size_t)st.st_size;
    result->owns_data = 0;

    result->data = (float *)((unsigned char *)mapped + TENSOR_DATA_OFFSET);

    *out = result;

    return TF_OK;
}