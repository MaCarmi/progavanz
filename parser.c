/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "parser.h"
#include "error.h"

void token_free(Token *tok) {
    if (!tok) return;
    if (tok->type == TOKEN_TENSOR && tok->as.tensor) {
        tensor_release(tok->as.tensor);
        tok->as.tensor = NULL;
    } else if (tok->type == TOKEN_STRING && tok->as.str) {
        free(tok->as.str);
        tok->as.str = NULL;
    }
}

/* Salta spazi e ritorni a capo */
static void skip_whitespace(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!isspace(c)) {
            ungetc(c, fp);
            break;
        }
    }
}

/* Parsing tensori 1D: formato [ 1 2 3 ] con spazi obbligatori */
static Tensor *parse_tensor_literal(FILE *fp) {
    // Verifichiamo lo spazio obbligatorio dopo '['
    int next = fgetc(fp);
    if (next != ' ' && next != '\t' && next != '\n' && next != '\r') {
        error_fatal(ERR_SYNTAX_ERROR, "Spazio obbligatorio mancante dopo '['");
    }

    size_t capacity = 8;
    size_t count = 0;
    float *values = (float *)malloc(capacity * sizeof(float));
    if (!values) error_fatal(ERR_OUT_OF_MEMORY, "Allocazione fallita durante il parsing del tensore");

    while (1) {
        skip_whitespace(fp);
        int c = fgetc(fp);

        if (c == EOF) {
            free(values);
            error_fatal(ERR_SYNTAX_ERROR, "Parentesi ']' mancante prima della fine del file");
        }

        if (c == ']') {
            break;
        }

        // Controllo sintattico: vietate le virgole
        if (c == ',') {
            free(values);
            error_fatal(ERR_SYNTAX_ERROR, "Trovata virgola non consentita nei tensori");
        }

        ungetc(c, fp);

        float val;
        if (fscanf(fp, "%f", &val) == 1) {
            if (count >= capacity) {
                capacity *= 2;
                float *tmp = (float *)realloc(values, capacity * sizeof(float));
                if (!tmp) {
                    free(values);
                    error_fatal(ERR_OUT_OF_MEMORY, "Realloc fallita per il tensore");
                }
                values = tmp;
            }
            values[count++] = val;
        } else {
            free(values);
            error_fatal(ERR_SYNTAX_ERROR, "Valore numerico non valido all'interno del tensore");
        }
    }

    if (count == 0) {
        free(values);
        error_fatal(ERR_SYNTAX_ERROR, "Impossibile creare un tensore vuoto []");
    }

    size_t shape[1] = { count };
    Tensor *t = tensor_create(shape, 1);
    if (!t) {
        free(values);
        error_fatal(ERR_OUT_OF_MEMORY, "Creazione del tensore fallita");
    }

    memcpy(t->data, values, count * sizeof(float));
    free(values);

    return t;
}

/* Parsing stringhe: "file.pgm" */
static char *parse_string_literal(FILE *fp) {
    size_t capacity = 32;
    size_t len = 0;
    char *buf = (char *)malloc(capacity);
    if (!buf) error_fatal(ERR_OUT_OF_MEMORY, "Allocazione memoria per la stringa fallita");

    int c;
    while ((c = fgetc(fp)) != EOF && c != '"') {
        if (len + 1 >= capacity) {
            capacity *= 2;
            char *tmp = (char *)realloc(buf, capacity);
            if (!tmp) {
                free(buf);
                error_fatal(ERR_OUT_OF_MEMORY, "Realloc per stringa fallita");
            }
            buf = tmp;
        }
        buf[len++] = (char)c;
    }

    if (c == EOF) {
        free(buf);
        error_fatal(ERR_SYNTAX_ERROR, "Doppio apice di chiusura '\"' mancante");
    }

    buf[len] = '\0';
    return buf;
}

/* Riconosce e restituisce il prossimo token */
Token parser_next_token(FILE *fp) {
    Token tok;
    memset(&tok, 0, sizeof(Token));

    skip_whitespace(fp);
    int c = fgetc(fp);

    if (c == EOF) {
        tok.type = TOKEN_EOF;
        return tok;
    }

    if (c == '[') {
        tok.type = TOKEN_TENSOR;
        tok.as.tensor = parse_tensor_literal(fp);
        return tok;
    }

    if (c == '"') {
        tok.type = TOKEN_STRING;
        tok.as.str = parse_string_literal(fp);
        return tok;
    }

    /* Operatore o identificatore singolo/multiplo */
    tok.type = TOKEN_OPERATOR;
    int idx = 0;
    tok.as.op_str[idx++] = (char)c;

    while ((c = fgetc(fp)) != EOF && !isspace(c) && c != '[' && c != '"') {
        if (idx < 15) {
            tok.as.op_str[idx++] = (char)c;
        }
    }

    if (c != EOF && (isspace(c) || c == '[' || c == '"')) {
        ungetc(c, fp);
    }

    tok.as.op_str[idx] = '\0';
    return tok;
}