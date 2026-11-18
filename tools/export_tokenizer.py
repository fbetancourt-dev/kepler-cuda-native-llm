#!/usr/bin/env python3
"""
export_tokenizer.py: Convierte models/tokenizer.json a models/tokenizer.bin
Formato binario ultraligero y zero-dependency para C.
"""

import json
import struct
import sys

def main():
    json_path = "models/tokenizer.json"
    bin_path = "models/tokenizer.bin"

    print(f"[*] Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]  # token_str -> id
    merges_raw = data["model"]["merges"]  # list of "token1 token2"

    vocab_size = len(vocab)
    num_merges = len(merges_raw)

    print(f"[+] Vocab size: {vocab_size}, Merges: {num_merges}")

    # Invertir vocabulario a id -> str
    id_to_token = [""] * vocab_size
    max_token_len = 0
    for tok_str, tok_id in vocab.items():
        id_to_token[tok_id] = tok_str
        tok_len = len(tok_str.encode("utf-8"))
        if tok_len > max_token_len:
            max_token_len = tok_len

    # Parsear merges a tuplas (id1, id2, merged_id)
    merges = []
    for merge_str in merges_raw:
        parts = merge_str.split(" ")
        if len(parts) != 2:
            continue
        p1, p2 = parts[0], parts[1]
        merged = p1 + p2
        if p1 in vocab and p2 in vocab and merged in vocab:
            merges.append((vocab[p1], vocab[p2], vocab[merged]))

    print(f"[+] Valid merges resolved: {len(merges)}")

    print(f"[*] Writing binary format to {bin_path}...")
    with open(bin_path, "wb") as f:
        # Header: MAGIC "TOK1", vocab_size, num_merges, max_token_len
        magic = b"TOK1"
        f.write(magic)
        f.write(struct.pack("<III", vocab_size, len(merges), max_token_len))

        # Vocabulario
        for i in range(vocab_size):
            b_str = id_to_token[i].encode("utf-8")
            f.write(struct.pack("<H", len(b_str)))
            f.write(b_str)

        # Merges
        for id1, id2, m_id in merges:
            f.write(struct.pack("<iii", id1, id2, m_id))

    print(f"[+] {bin_path} generated successfully! Size: {struct.calcsize('<III') + sum(2 + len(s.encode('utf-8')) for s in id_to_token) + len(merges)*12} bytes")

if __name__ == "__main__":
    main()
