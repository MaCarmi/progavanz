/*
Marco Carmignano
SM3201526
 */

#include "io_pgm.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
Converte un valore float in un valore unsigned char per la scrittura in un file PGM binario.

Restituisce in input: x che è il valore float da convertire
Restituisce in output: il valore unsigned char corrispondente, con valori limitati a [0, 255]
 */
static unsigned char float_to_pixel(float x)
{
    if (x < 0.0f) {
        x = 0.0f;
    }

    if (x > 1.0f) {
        x = 1.0f;
    }

    return (unsigned char)(x * 255.0f + 0.5f);
}


/*
Scrive un tensore 2D in un file PGM binario P5.

Restituisce in input: a che è il tensore da scrivere e filename che è il percorso del file PGM da creare
Restituisce in output: TF_OK se l'operazione è andata a buon fine, altrimenti un codice di errore
 */
TFStatus tf_write_pgm(const Tensor *a, const char *filename)
{
    if (a == NULL || filename == NULL) {
        return TF_ERR_NULL_ARGUMENT;
    }

    if (a->ndim != 2) {
        return TF_ERR_DIMENSION_MISMATCH;
    }

    const size_t height = a->shape[0];
    const size_t width = a->shape[1];

    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        return TF_ERR_FILE;
    }

    if (fprintf(fp, "P5\n%zu %zu\n255\n", width, height) < 0) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    unsigned char *pixels =
        (unsigned char *)malloc(a->total_size * sizeof(unsigned char));

    if (pixels == NULL) {
        fclose(fp);
        return TF_ERR_ALLOCATION;
    }

    for (size_t i = 0; i < a->total_size; i++) {
        pixels[i] = float_to_pixel(a->data[i]);
    }

    size_t written = fwrite(
        pixels,
        sizeof(unsigned char),
        a->total_size,
        fp
    );

    free(pixels);

    if (written != a->total_size) {
        fclose(fp);
        return TF_ERR_FILE;
    }

    if (fclose(fp) != 0) {
        return TF_ERR_FILE;
    }

    return TF_OK;
}