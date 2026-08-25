/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef IO_TENSOR_H
#define IO_TENSOR_H

#include "tensor.h"
#include "tf_status.h"

/*
Legge un tensore da un file nel formato TensorForth.

Prende in input: filename che è il percorso del file da leggere
Restituisce in output: out che è il puntatore al tensore risultato della lettura, che contiene i valori del tensore letto
 */
TFStatus tf_read_tensor_mmap(const char *filename, Tensor **out);


/*
Salva un tensore in un file nel formato TensorForth.

Prende in input: a che è il tensore da salvare e filename che è il percorso del file da creare
Restituisce in output: TF_OK se l'operazione è andata a buon fine, altrimenti un codice di errore
 */
TFStatus tf_write_tensor_file(const Tensor *a, const char *filename);

#endif /* IO_TENSOR_H */