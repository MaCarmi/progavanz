#ifndef STACK_H
#define STACK_H

#include "tensor.h"

#define INITIAL_STACK_CAPACITY 16

typedef struct {
    Value **data;      // Array di puntatori a Value
    int top;           // Indice dell'elemento in cima (-1 se vuoto)
    int capacity;      // Capacità massima attuale
} Stack;

// Funzioni di gestione ciclo di vita
Stack *stack_create(void);
void stack_free(Stack *s);

// Operazioni fondamentali dello stack (Forth Primitives)
int stack_push(Stack *s, Value *v);
Value *stack_pop(Stack *s);
Value *stack_peek(const Stack *s);

// Utility e manipolazione Forth
void stack_dup(Stack *s);
void stack_drop(Stack *s);
void stack_swap(Stack *s);
void stack_rot(Stack *s);
void stack_print(const Stack *s);

#endif // STACK_H