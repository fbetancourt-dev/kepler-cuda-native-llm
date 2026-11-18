#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ====================================================================
// ESTRUCTURAS DE DATOS INTERNAS
// ====================================================================

typedef struct {
    int32_t id1;
    int32_t id2;
    int32_t merged_id;
    int32_t rank;
} merge_entry_t;

typedef struct {
    int32_t id1;
    int32_t id2;
    int32_t merged_id;
    int32_t rank;
    uint8_t occupied;
} hash_slot_t;

// Tabla Hash de búsqueda de tokens iniciales (string -> token_id)
typedef struct {
    char *str;
    int32_t id;
    uint8_t occupied;
} token_hash_slot_t;

struct tokenizer {
    int32_t vocab_size;
    int32_t num_merges;
    int32_t max_token_len;
    char **vocab;               // Array de strings por ID: vocab[id]
    int32_t byte_pieces[256];   // IDs de <0x00> .. <0xFF>

    // Tabla Hash para Merges (id1, id2) -> (merged_id, rank)
    size_t hash_size;
    hash_slot_t *merge_table;

    // Tabla Hash para búsqueda rápida de subwords UTF-8 (str -> id)
    size_t tok_hash_size;
    token_hash_slot_t *tok_table;

    // Buffer de decodificación reutilizable
    char decode_buf[512];
};

// ====================================================================
// FUNCIONES HASH RAPIDAS
// ====================================================================

static inline uint64_t hash_pair(int32_t id1, int32_t id2) {
    uint64_t key = ((uint64_t)id1 << 32) | (uint32_t)id2;
    key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
    return key ^ (key >> 31);
}

static inline uint32_t hash_str(const char *str, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= (uint8_t)str[i];
        hash *= 16777619u;
    }
    return hash;
}

static void insert_merge(tokenizer_t *tok, int32_t id1, int32_t id2, int32_t m_id, int32_t rank) {
    uint64_t h = hash_pair(id1, id2);
    size_t mask = tok->hash_size - 1;
    size_t idx = (size_t)(h & mask);

    while (tok->merge_table[idx].occupied) {
        idx = (idx + 1) & mask;
    }

    tok->merge_table[idx].id1 = id1;
    tok->merge_table[idx].id2 = id2;
    tok->merge_table[idx].merged_id = m_id;
    tok->merge_table[idx].rank = rank;
    tok->merge_table[idx].occupied = 1;
}

static inline int find_merge(tokenizer_t *tok, int32_t id1, int32_t id2, int32_t *out_merged, int32_t *out_rank) {
    uint64_t h = hash_pair(id1, id2);
    size_t mask = tok->hash_size - 1;
    size_t idx = (size_t)(h & mask);

    while (tok->merge_table[idx].occupied) {
        if (tok->merge_table[idx].id1 == id1 && tok->merge_table[idx].id2 == id2) {
            *out_merged = tok->merge_table[idx].merged_id;
            *out_rank = tok->merge_table[idx].rank;
            return 1;
        }
        idx = (idx + 1) & mask;
    }
    return 0;
}

static void insert_token_str(tokenizer_t *tok, const char *str, int32_t id) {
    int len = strlen(str);
    uint32_t h = hash_str(str, len);
    size_t mask = tok->tok_hash_size - 1;
    size_t idx = (size_t)(h & mask);

    while (tok->tok_table[idx].occupied) {
        idx = (idx + 1) & mask;
    }

    tok->tok_table[idx].str = strdup(str);
    tok->tok_table[idx].id = id;
    tok->tok_table[idx].occupied = 1;
}

static inline int find_token_str(tokenizer_t *tok, const char *str, int len, int32_t *out_id) {
    uint32_t h = hash_str(str, len);
    size_t mask = tok->tok_hash_size - 1;
    size_t idx = (size_t)(h & mask);

    while (tok->tok_table[idx].occupied) {
        if (strncmp(tok->tok_table[idx].str, str, len) == 0 && tok->tok_table[idx].str[len] == '\0') {
            *out_id = tok->tok_table[idx].id;
            return 1;
        }
        idx = (idx + 1) & mask;
    }
    return 0;
}

// ====================================================================
// CARGA DEL TOKENIZER
// ====================================================================

tokenizer_t *tokenizer_init(const char *bin_path) {
    FILE *f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "[-] tokenizer_init: Cannot open '%s'\n", bin_path);
        return NULL;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TOK1", 4) != 0) {
        fprintf(stderr, "[-] tokenizer_init: Invalid magic header in '%s'\n", bin_path);
        fclose(f);
        return NULL;
    }

    uint32_t vocab_size, num_merges, max_token_len;
    if (fread(&vocab_size, sizeof(uint32_t), 1, f) != 1 ||
        fread(&num_merges, sizeof(uint32_t), 1, f) != 1 ||
        fread(&max_token_len, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "[-] tokenizer_init: Failed to read header\n");
        fclose(f);
        return NULL;
    }

    tokenizer_t *tok = (tokenizer_t *)calloc(1, sizeof(tokenizer_t));
    tok->vocab_size = (int32_t)vocab_size;
    tok->num_merges = (int32_t)num_merges;
    tok->max_token_len = (int32_t)max_token_len;

    tok->vocab = (char **)malloc(vocab_size * sizeof(char *));

    // Tabla Hash para Tokens (65536 slots)
    tok->tok_hash_size = 65536;
    tok->tok_table = (token_hash_slot_t *)calloc(tok->tok_hash_size, sizeof(token_hash_slot_t));

    // Cargar vocabulario
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) break;
        char *str = (char *)malloc(len + 1);
        if (fread(str, 1, len, f) != len) { free(str); break; }
        str[len] = '\0';
        tok->vocab[i] = str;
        insert_token_str(tok, str, (int32_t)i);

        // Detectar tokens de byte <0xXX> (IDs 3..258)
        if (len == 6 && str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
            unsigned int bval = 0;
            if (sscanf(str + 3, "%02x", &bval) == 1 && bval < 256) {
                tok->byte_pieces[bval] = (int32_t)i;
            }
        }
    }

    // Tabla Hash para Merges (262144 slots para 61249 merges, ~23% carga)
    tok->hash_size = 262144;
    tok->merge_table = (hash_slot_t *)calloc(tok->hash_size, sizeof(hash_slot_t));

    // Cargar merges
    for (uint32_t i = 0; i < num_merges; i++) {
        int32_t id1, id2, m_id;
        if (fread(&id1, sizeof(int32_t), 1, f) != 1 ||
            fread(&id2, sizeof(int32_t), 1, f) != 1 ||
            fread(&m_id, sizeof(int32_t), 1, f) != 1) break;

        insert_merge(tok, id1, id2, m_id, (int32_t)i);
    }

    fclose(f);
    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    if (tok->tok_table) {
        for (size_t i = 0; i < tok->tok_hash_size; i++) {
            if (tok->tok_table[i].str) free(tok->tok_table[i].str);
        }
        free(tok->tok_table);
    }
    if (tok->merge_table) free(tok->merge_table);
    free(tok);
}

// ====================================================================
// CODIFICACIÓN BPE
// ====================================================================

int tokenizer_encode(tokenizer_t *tok, const char *text, int8_t bos, int8_t eos, int32_t *tokens, int max_tokens) {
    if (!tok || !text || !tokens || max_tokens <= 0) return -1;

    int num_tokens = 0;
    if (bos && num_tokens < max_tokens) {
        tokens[num_tokens++] = TOKEN_BOS;
    }

    if (*text == '\0') {
        if (eos && num_tokens < max_tokens) {
            tokens[num_tokens++] = TOKEN_EOS;
        }
        return num_tokens;
    }

    // Buffer temporal para texto procesado con prefijo ' ' (U+2581 = 0xE2 0x96 0x81)
    size_t text_len = strlen(text);
    char *prefixed = (char *)malloc(text_len * 3 + 16);
    size_t pref_idx = 0;

    // SentencePiece añade ' ' al inicio si no empieza con espacio
    if ((uint8_t)text[0] != ' ') {
        prefixed[pref_idx++] = (char)0xe2;
        prefixed[pref_idx++] = (char)0x96;
        prefixed[pref_idx++] = (char)0x81;
    }

    for (size_t i = 0; i < text_len; i++) {
        if ((uint8_t)text[i] == ' ') {
            prefixed[pref_idx++] = (char)0xe2;
            prefixed[pref_idx++] = (char)0x96;
            prefixed[pref_idx++] = (char)0x81;
        } else {
            prefixed[pref_idx++] = text[i];
        }
    }
    prefixed[pref_idx] = '\0';

    // Descomponer en tokens iniciales
    int32_t *seq = (int32_t *)malloc((pref_idx + 8) * sizeof(int32_t));
    int seq_len = 0;

    const char *p = prefixed;
    while (*p) {
        int char_len = 1;
        uint8_t c = (uint8_t)*p;
        if ((c & 0x80) == 0x00)      char_len = 1;
        else if ((c & 0xe0) == 0xc0) char_len = 2;
        else if ((c & 0xf0) == 0xe0) char_len = 3;
        else if ((c & 0xf8) == 0xf0) char_len = 4;

        int32_t tid;
        if (find_token_str(tok, p, char_len, &tid)) {
            seq[seq_len++] = tid;
        } else {
            // Byte fallback para bytes individuales
            for (int b = 0; b < char_len; b++) {
                uint8_t b_val = (uint8_t)p[b];
                seq[seq_len++] = tok->byte_pieces[b_val];
            }
        }
        p += char_len;
    }
    free(prefixed);

    // Bucle BPE: fusionar el par con menor rank sucesivamente
    while (seq_len >= 2) {
        int best_idx = -1;
        int32_t best_rank = 0x7fffffff;
        int32_t best_merged = -1;

        for (int i = 0; i < seq_len - 1; i++) {
            int32_t m_id, rank;
            if (find_merge(tok, seq[i], seq[i + 1], &m_id, &rank)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_merged = m_id;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) break; // No hay más merges aplicables

        // Aplicar el mejor merge
        seq[best_idx] = best_merged;
        for (int i = best_idx + 1; i < seq_len - 1; i++) {
            seq[i] = seq[i + 1];
        }
        seq_len--;
    }

    // Copiar a la salida
    for (int i = 0; i < seq_len && num_tokens < max_tokens; i++) {
        tokens[num_tokens++] = seq[i];
    }
    free(seq);

    if (eos && num_tokens < max_tokens) {
        tokens[num_tokens++] = TOKEN_EOS;
    }

    return num_tokens;
}

// ====================================================================
// DECODIFICACIÓN
// ====================================================================

const char *tokenizer_decode(tokenizer_t *tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }

    const char *raw = tok->vocab[token_id];
    size_t raw_len = strlen(raw);

    // Byte fallback token: <0xXX>
    if (raw_len == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        unsigned int bval = 0;
        if (sscanf(raw + 3, "%02x", &bval) == 1) {
            tok->decode_buf[0] = (char)bval;
            tok->decode_buf[1] = '\0';
            return tok->decode_buf;
        }
    }

    // Reemplazar prefijo SentencePiece ' ' (0xE2 0x96 0x81) por espacio normal
    size_t out_idx = 0;
    for (size_t i = 0; i < raw_len && out_idx < sizeof(tok->decode_buf) - 2; ) {
        if ((uint8_t)raw[i] == 0xe2 && (uint8_t)raw[i+1] == 0x96 && (uint8_t)raw[i+2] == 0x81) {
            tok->decode_buf[out_idx++] = ' ';
            i += 3;
        } else {
            tok->decode_buf[out_idx++] = raw[i++];
        }
    }
    tok->decode_buf[out_idx] = '\0';
    return tok->decode_buf;
}

// ====================================================================
// FORMATEO DE PROMPT (PLANTILLA CHATML / ZEPHYR / TINYLLAMA)
// ====================================================================

void tokenizer_format_chat_prompt(char *out_buf, size_t max_len, const char *user_prompt) {
    snprintf(out_buf, max_len,
        "<|system|>\n"
        "You are a helpful and concise AI assistant running on Kepler silicon.</s>\n"
        "<|user|>\n"
        "%s</s>\n"
        "<|assistant|>\n",
        user_prompt ? user_prompt : "Hello");
}
