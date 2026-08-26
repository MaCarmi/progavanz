$ErrorActionPreference = "Stop"

$sources = @(
    "main.c",
    "parser.c",
    "stack.c",
    "tensor.c",
    "error.c",
    "ops_elementwise.c",
    "ops_matrix.c",
    "ops_convolution.c",
    "io_pgm.c",
    "io_tensor.c"
)

Write-Host "Compilo tensorforth.exe"
gcc -Wall -Wextra -std=c11 -O2 -fopenmp @sources -o tensorforth.exe
if ($LASTEXITCODE -ne 0) {
    throw "Compilazione fallita"
}

New-Item -ItemType Directory -Force examples | Out-Null

$okTests = @(
    @{
        File = "tests/01_print.tf"
        Title = "Stampa base"
        What = "Crea un tensore 1D e lo stampa."
        Expected = "Deve mostrare shape=[3] e data=[1.00 2.00 3.00]."
    },
    @{
        File = "tests/02_elementwise.tf"
        Title = "Operazioni elemento per elemento"
        What = "Testa somma, sottrazione, prodotto, ReLU e somma totale."
        Expected = "Devono uscire 5 tensori; la sottrazione segue il PDF: ( b a -- a-b )."
    },
    @{
        File = "tests/03_comparison_logic_select.tf"
        Title = "Confronti, logica e select"
        What = "Testa <, >, =, &, \|, ! e selezione con maschera $."
        Expected = "I confronti/logici producono 0.00 o 1.00; select restituisce [100.00 2.00 100.00]."
    },
    @{
        File = "tests/04_shape_reshape_ravel_fill.tf"
        Title = "Shape, reshape, ravel e fill"
        What = "Crea matrici con f, cambia forma con r, legge la shape con # e appiattisce con _."
        Expected = "Devono comparire shape=[2 3], shape=[2], shape=[6] e shape=[2 2]."
    },
    @{
        File = "tests/05_matrix_ops.tf"
        Title = "Prodotto scalare e matriciale"
        What = "Testa . su due vettori e @ su due matrici."
        Expected = "Il prodotto scalare vale 32; il prodotto matriciale A@B ha shape=[2 2]."
    },
    @{
        File = "tests/06_convolution.tf"
        Title = "Convoluzione 2D"
        What = "Applica un kernel identita 3x3 a una matrice 3x3."
        Expected = "Il risultato deve avere shape=[3 3] e stessi valori della matrice iniziale."
    },
    @{
        File = "tests/07_stack.tf"
        Title = "Operatori stack"
        What = "Testa dup d, swap s, over o e drop D insieme ad alcune somme/sottrazioni."
        Expected = "Devono uscire tre tensori risultato senza errori di stack."
    },
    @{
        File = "tests/08_tensor_io.tf"
        Title = "I/O formato TensorForth"
        What = "Scrive un tensore su file con } e lo rilegge con {."
        Expected = "Il tensore letto deve essere shape=[2 2] con data=[1.00 2.00 3.00 4.00]."
    },
    @{
        File = "tests/09_pgm_io.tf"
        Title = "I/O immagine PGM"
        What = "Scrive un tensore 2D come PGM con ) e lo rilegge con (."
        Expected = "Il tensore letto deve avere shape=[2 3]; il valore 1.2 viene limitato a 1.00."
    },
    @{
        File = "tests/10_random.tf"
        Title = "Tensore casuale"
        What = "Crea un tensore casuale con ? usando shape=[2 2]."
        Expected = "Deve mostrare shape=[2 2] con valori casuali tra 0.00 e 1.00."
    },
    @{
        File = "tests/13_pdf_duplicate.tf"
        Title = "Esempio PDF: duplica tensore"
        What = "Esegue il programma del PDF: [ 5 5 ] d + p."
        Expected = "Deve stampare Tensor(shape=[2], data=[10.00 10.00])."
    },
    @{
        File = "tests/14_pdf_matrix_multiplication.tf"
        Title = "Esempio PDF: matrix multiplication"
        What = "Esegue il programma del PDF con due tensori random, reshape e @."
        Expected = "Deve stampare una matrice con shape=[3 5]. I valori sono casuali."
    },
    @{
        File = "tests/15_prepare_blur_input.tf"
        Title = "Setup blur PGM"
        What = "Crea examples/cray-2.pgm usando l'operatore ) per preparare l'esempio del PDF."
        Expected = "Non stampa nulla; deve creare il file PGM di input senza errori."
    },
    @{
        File = "tests/16_pdf_blur.tf"
        Title = "Esempio PDF: blurring immagine"
        What = "Legge examples/cray-2.pgm, applica kernel 5x5 pieno di 0.04, scrive e rilegge l'output."
        Expected = "Deve stampare il PGM sfocato come tensore con shape=[4 4]."
    }
)

foreach ($test in $okTests) {
    Write-Host ""
    Write-Host "=== $($test.File) - $($test.Title) ==="
    Write-Host "Cosa fa: $($test.What)"
    Write-Host "Atteso:  $($test.Expected)"
    Write-Host "Output:"
    .\tensorforth.exe $test.File
    if ($LASTEXITCODE -ne 0) {
        throw "Test fallito: $($test.File)"
    }
}

$errorTests = @(
    @{
        File = "tests/11_error_underflow.tf"
        Title = "Errore stack underflow"
        What = "Prova a usare . senza i due tensori richiesti."
        Expected = "Deve stampare RUN_TIME ERROR [1]: Stack Underflow."
    },
    @{
        File = "tests/12_error_dim_mismatch.tf"
        Title = "Errore dimension mismatch"
        What = "Prova a sommare due tensori con shape diversa."
        Expected = "Deve stampare RUN_TIME ERROR [3]: Dimension Mismatch."
    },
    @{
        File = "tests/17_error_type_mismatch.tf"
        Title = "Errore type mismatch"
        What = "Passa una stringa all'operatore p, che richiede un tensore."
        Expected = "Deve stampare RUN_TIME ERROR [2]: Type Mismatch."
    },
    @{
        File = "tests/18_error_file_not_found.tf"
        Title = "Errore file non trovato"
        What = "Prova a leggere un PGM inesistente."
        Expected = "Deve stampare RUN_TIME ERROR [5]: File Error."
    },
    @{
        File = "tests/19_error_syntax_tensor.tf"
        Title = "Errore sintassi tensore"
        What = "Usa [1 2 3] senza spazio dopo [, forma vietata dal PDF."
        Expected = "Deve stampare RUN_TIME ERROR [4]: Syntax Error."
    }
)

foreach ($test in $errorTests) {
    Write-Host ""
    Write-Host "=== Errore atteso: $($test.File) - $($test.Title) ==="
    Write-Host "Cosa fa: $($test.What)"
    Write-Host "Atteso:  $($test.Expected)"
    Write-Host "Output:"
    .\tensorforth.exe $test.File
    if ($LASTEXITCODE -eq 0) {
        throw "Il test doveva fallire ma e' passato: $($test.File)"
    }
    Write-Host "OK: errore rilevato come previsto"
}

Write-Host ""
Write-Host "Tutti i test sono completati."
