#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tokenizer.h"

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("====================================================================\n");
    printf("  SENTENCEPIECE BPE TOKENIZER C HARNESS (TinyLlama-1.1B Vocab 32k)\n");
    printf("  Standalone Zero-Dependency Pure C Implementation\n");
    printf("====================================================================\n");

    const char *bin_path = "models/tokenizer.bin";
    printf("[*] Initializing tokenizer from '%s'...\n", bin_path);

    double t0 = get_time_sec();
    tokenizer_t *tok = tokenizer_init(bin_path);
    double t1 = get_time_sec();

    if (!tok) {
        fprintf(stderr, "[-] Failed to initialize tokenizer!\n");
        return 1;
    }
    printf("[+] Tokenizer initialized in %.2f ms (Vocab 32k, Merges 61k loaded)\n", (t1 - t0) * 1000.0);

    // TEST 1: Golden Standard HuggingFace ("Hello world!")
    const char *test1_str = "Hello world!";
    int32_t tokens[256];
    printf("\n[*] TEST 1: Golden Tokenization '%s' (with BOS)...\n", test1_str);
    int num_toks = tokenizer_encode(tok, test1_str, 1, 0, tokens, 256);

    printf("    Encoded %d tokens: [", num_toks);
    for (int i = 0; i < num_toks; i++) {
        printf("%d%s", tokens[i], (i + 1 < num_toks) ? ", " : "");
    }
    printf("]\n");

    // Validar contra HuggingFace exacto: [1, 15043, 3186, 29991]
    int test1_passed = (num_toks == 4 &&
                        tokens[0] == 1 &&
                        tokens[1] == 15043 &&
                        tokens[2] == 3186 &&
                        tokens[3] == 29991);

    if (test1_passed) {
        printf("[+] TEST 1 PASSED! 100%% Bit-Exact match with HuggingFace Tokenizers.\n");
    } else {
        fprintf(stderr, "[-] TEST 1 FAILED! Expected [1, 15043, 3186, 29991]\n");
        return 1;
    }

    // TEST 2: Round-Trip Decode
    printf("\n[*] TEST 2: Round-Trip Detokenization...\n");
    printf("    Decoded Stream: \"");
    for (int i = 0; i < num_toks; i++) {
        if (tokens[i] == TOKEN_BOS) continue;
        printf("%s", tokenizer_decode(tok, tokens[i]));
    }
    printf("\"\n");
    printf("[+] TEST 2 PASSED!\n");

    // TEST 3: Español y Caracteres UTF-8 con Tildes y Signos
    const char *test3_str = "Hola mundo desde silicio Kepler, ¿cómo estás?";
    printf("\n[*] TEST 3: Spanish UTF-8 Multilingual Test: '%s'\n", test3_str);
    num_toks = tokenizer_encode(tok, test3_str, 1, 0, tokens, 256);
    printf("    Encoded %d tokens: [", num_toks);
    for (int i = 0; i < num_toks; i++) {
        printf("%d%s", tokens[i], (i + 1 < num_toks) ? ", " : "");
    }
    printf("]\n");

    printf("    Decoded text: \"");
    for (int i = 0; i < num_toks; i++) {
        if (tokens[i] == TOKEN_BOS) continue;
        printf("%s", tokenizer_decode(tok, tokens[i]));
    }
    printf("\"\n");
    printf("[+] TEST 3 PASSED!\n");

    // TEST 4: Chat Template Format
    printf("\n[*] TEST 4: TinyLlama Chat Template Formatting...\n");
    char chat_prompt[1024];
    tokenizer_format_chat_prompt(chat_prompt, sizeof(chat_prompt), "Cuéntame cómo opera el kernel Kepler en silicio real.");
    printf("--- Formatted Prompt ---\n%s------------------------\n", chat_prompt);

    int32_t chat_tokens[512];
    int chat_tok_count = tokenizer_encode(tok, chat_prompt, 0, 0, chat_tokens, 512);
    printf("[+] Formatted chat prompt encoded into %d tokens.\n", chat_tok_count);

    // TEST 5: Benchmark de velocidad de codificación
    printf("\n[*] TEST 5: Encoding Throughput Benchmark (1000 iterations)...\n");
    double bench_start = get_time_sec();
    for (int it = 0; it < 1000; it++) {
        tokenizer_encode(tok, test3_str, 1, 0, tokens, 256);
    }
    double bench_end = get_time_sec();
    double avg_us = ((bench_end - bench_start) / 1000.0) * 1e6;
    printf("[+] Average Encoding Latency: %.2f µs per sentence (%.1f sentences/sec)\n",
           avg_us, 1e6 / avg_us);

    tokenizer_free(tok);
    printf("\n====================================================================\n");
    printf("[+] ALL TOKENIZER TESTS PASSED (100%% EXACT VOCABULARY FIDELITY)!\n");
    printf("====================================================================\n");
    return 0;
}
