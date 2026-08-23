/*
Marco Carmignano
SM3201526
*/

#ifndef OPS_ELEMENTWISE_H
#define OPS_ELEMENTWISE_H

#include "tensor.h"
#include "tf_status.h"

/*
Operazioni aritmetiche elemento per elemento
Richiedono che i tensori abbiano le stesse dimensioni
*/
TFStatus tf_add(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_sub(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_mul(const Tensor *a, const Tensor *b, Tensor **out);


/*
Operazioni di comparazione elemento per elemento
Restituiscono in output 1.0f se vero o 0.0f se falso
*/
TFStatus tf_less(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_greater(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_equal(const Tensor *a, const Tensor *b, Tensor **out);

/*
Operazioni logiche elemento per elemento
*/
TFStatus tf_and(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_or(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_not(const Tensor *a, Tensor **out);

/*
Operazioni di selezione 

Operano in questo modo:
mask[i] == 1.0f -> out[i] = a[i]
mask[i] == 0.0f -> out[i] = b[i]
*/
TFStatus tf_select(
    const Tensor *b,
    const Tensor *a,
    const Tensor *mask,
    Tensor **out
);

/*
Operazioni elemento per elemento
*/
TFStatus tf_relu(const Tensor *a, Tensor **out);
TFStatus tf_min(const Tensor *a, const Tensor *b, Tensor **out);
TFStatus tf_max(const Tensor *a, const Tensor *b, Tensor **out);


/*
Operazione di riduzione
Ritorna un tensore 1D con un solo elemento contenente la somma di tutti gli elementi di a
 */
TFStatus tf_sum(const Tensor *a, Tensor **out);

#endif