/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */
 
#ifndef IO_PGM_H
#define IO_PGM_H

#include "tensor.h"
#include "tf_status.h"

/*
Legge un'immagine PGM binaria P5 e la converte in un tensore 2D.

Restituisce in input: filename che è il percorso del file PGM da leggere
Restituisce in output: out che è il puntatore al tensore risultato della lettura, che contiene i valori dei pixel normalizzati in [0, 1]
 */
TFStatus tf_read_pgm(const char *filename, Tensor **out);


/*
Scrive un tensore 2D in un file PGM binario P5.

Restituisce in input: a che è il tensore da scrivere e filename che è il percorso del file PGM da creare
Restituisce in output: TF_OK se l'operazione è andata a buon fine
 */
TFStatus tf_write_pgm(const Tensor *a, const char *filename);

#endif /* IO_PGM_H */