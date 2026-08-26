/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef IO_TENSOR_H
#define IO_TENSOR_H

#include "tensor.h"
#include "error.h"

/*
 * Legge un tensore da un file nel formato TensorForth usand mmap.
 *
 * Prende in input: filename che è il percorso del file da leggere
 * Restituisce in output: out che è il puntatore al tensore risultato della lettura
 * Restituisce: ErrorCode (ERR_NONE in caso di successo)
 */
ErrorCode tf_read_tensor_mmap(const char *filename, Tensor **out);

/*
 * Salva un tensore in un file nel formato TensorForth.
 *
 * Prende in input: a che è il tensore da salvare e filename che è il percorso del file da creare
 * Restituisce: ErrorCode (ERR_NONE in caso di successo)
 */
ErrorCode tf_write_tensor_file(const Tensor *a, const char *filename);

#endif /* IO_TENSOR_H */
