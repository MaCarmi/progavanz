/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef IO_PGM_H
#define IO_PGM_H

#include "tensor.h"
#include "error.h"

/*
 * Legge un'immagine PGM binaria P5 e la converte in un tensore 2D.
 *
 * Parametri in input: filename - il percorso del file PGM da leggere
 * Parametri in output: out - puntatore al tensore risultato della lettura
 * Restituisce: ErrorCode (ERR_NONE in caso di successo)
 */
ErrorCode tf_read_pgm(const char *filename, Tensor **out);

/*
 * Scrive un tensore 2D in un file PGM binario P5.
 *
 * Parametri in input: a - il tensore da scrivere
 *                     filename - il percorso del file PGM da creare
 * Restituisce: ErrorCode (ERR_NONE in caso di successo)
 */
ErrorCode tf_write_pgm(const Tensor *a, const char *filename);

#endif /* IO_PGM_H */