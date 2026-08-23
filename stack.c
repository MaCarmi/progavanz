#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

Stack *stack_create(void) {
    Stack *s = (Stack *)malloc(sizeof(Stack));
    if (!s) return NULL;

    s->capacity = INITIAL_STACK_CAPACITY;
    s->top = -1;
    s->data = (Value **)malloc(s->capacity * sizeof(Value *));
    if (!s->data) {
        free(s);
        return NULL;
    }

    return s;
}

void stack_free(Stack *s) {
    if (!s) return;
    for (int i = 0; i <= s->top; i++) {
        value_release(s->data[i]);
    }
    free(s->data);
    free(s);
}

int stack_push(Stack *s, Value *v) {
    if (!s || !v) return 0;

    // Ridimensionamento dinamico se lo stack è pieno
    if (s->top + 1 >= s->capacity) {
        int new_cap = s->capacity * 2;
        Value **tmp = (Value **)realloc(s->data, new_cap * sizeof(Value *));
        if (!tmp) return 0;
        s->data = tmp;
        s->capacity = new_cap;
    }

    s->top++;
    s->data[s->top] = v;
    return 1;
}

Value *stack_pop(Stack *s) {
    if (!s || s->top < 0) return NULL;
    Value *v = s->data[s->top];
    s->top--;
    return v;
}

Value *stack_peek(const Stack *s) {
    if (!s || s->top < 0) return NULL;
    return s->data[s->top];
}

// Forth DUP: Duplica l'elemento in cima (incrementa ref_count via retain)
void stack_dup(Stack *s) {
    Value *top_val = stack_peek(s);
    if (!top_val) return;
    value_retain(top_val);
    stack_push(s, top_val);
}

// Forth DROP: Rimuove l'elemento in cima e decremente la memoria
void stack_drop(Stack *s) {
    Value *v = stack_pop(s);
    if (v) {
        value_release(v);
    }
}

// Forth SWAP: Scambia i primi due elementi in cima allo stack
void stack_swap(Stack *s) {
    if (!s || s->top < 1) return;
    Value *tmp = s->data[s->top];
    s->data[s->top] = s->data[s->top - 1];
    s->data[s->top - 1] = tmp;
}

// Forth ROT: Ruota i primi 3 elementi (portando il terzo in cima)
void stack_rot(Stack *s) {
    if (!s || s->top < 2) return;
    Value *third = s->data[s->top - 2];
    s->data[s->top - 2] = s->data[s->top - 1];
    s->data[s->top - 1] = s->data[s->top];
    s->data[s->top] = third;
}

void stack_print(const Stack *s) {
    if (!s) return;
    printf("STACK (%d/%d): [ ", s->top + 1, s->capacity);
    for (int i = 0; i <= s->top; i++) {
        value_print(s->data[i]);
        printf(" ");
    }
    printf("] <- TOP\n");
}