#!/usr/bin/env bash
set -e
export LD_LIBRARY_PATH="$(pwd)/bin:$LD_LIBRARY_PATH"

echo "========================================================================"
echo "  FULL SUITE REGRESSION: KEPLER GT 750M NATIVE CUDA DRIVER & LLM STACK"
echo "========================================================================"

SUITES=(
    "test_fase_a_hardening"
    "test_fase_b_runtime_api"
    "test_fase_c_streams_events"
    "test_mixed_args"
    "test_shared_reduction"
    "test_local_spill"
    "test_constant_memory"
    "test_sgemm_bench"
    "test_rms_norm"
    "test_gemv"
    "test_rope_kcache"
    "test_attention_decode"
    "test_swiglu"
    "test_transformer_layer"
    "test_lm_head"
    "test_transformer_robustness"
    "test_gemv_q8_0"
    "test_tokenizer"
    "test_transformer_layer_q8_0"
    "test_tinyllama_runner"
)

PASS_COUNT=0
TOTAL_COUNT=${#SUITES[@]}

for suite in "${SUITES[@]}"; do
    echo ""
    echo ">>> Running $suite..."
    ./bin/$suite
    echo "[PASS] $suite completed successfully."
    PASS_COUNT=$((PASS_COUNT + 1))
done

echo ""
echo "========================================================================"
echo "  ALL $PASS_COUNT / $TOTAL_COUNT SUITES PASSED ON PHYSICAL SILICON (0 ERRORS)!"
echo "========================================================================"
