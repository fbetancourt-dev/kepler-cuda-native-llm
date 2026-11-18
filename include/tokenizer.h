#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Token IDs especiales estándar de LLaMA / TinyLlama
#define TOKEN_UNK 0
#define TOKEN_BOS 1
#define TOKEN_EOS 2

typedef struct tokenizer tokenizer_t;

// Carga el modelo binario de tokenizer
tokenizer_t *tokenizer_init(const char *bin_path);

// Libera los recursos del tokenizer
void tokenizer_free(tokenizer_t *tok);

// Codifica un texto en una secuencia de tokens
// bos = 1 para anteponer <s> (ID 1), eos = 1 para posponer </s> (ID 2)
// Retorna el número de tokens codificados (o -1 si hay error)
int tokenizer_encode(tokenizer_t *tok, const char *text, int8_t bos, int8_t eos, int32_t *tokens, int max_tokens);

// Decodifica un token a su representación en texto UTF-8
const char *tokenizer_decode(tokenizer_t *tok, int32_t token_id);

// Formatea un prompt de usuario con la plantilla de chat de TinyLlama / Zephyr
void tokenizer_format_chat_prompt(char *out_buf, size_t max_len, const char *user_prompt);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H
