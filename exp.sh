#!/usr/bin/env bash
set -euo pipefail

BIN=./hw4
TEST_DIR=testcases
OUT_DIR=outputs

mkdir -p "$OUT_DIR"

in_file="$TEST_DIR/case01.in"
exp_file="$TEST_DIR/case01.out"
out_file="$OUT_DIR/run_case01.out"

if [[ ! -f "$in_file" ]]; then
    echo "Missing input file: $in_file"
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "Binary $BIN not executable."
    exit 1
fi

# Measure true runtime of the binary
echo "Running case01..."

time_output=$(mktemp)
bash -c "TIMEFORMAT='%3R'; time $BIN '$in_file' '$out_file' >/dev/null 2>&1" 2> "$time_output"
elapsed=$(cat "$time_output" | tr -d '\r\n')
rm -f "$time_output"

elapsed=$(printf "%.2f" "$elapsed")

# Read nonce from output 2nd line
produced="NA"
expected="NA"

if [[ -f "$out_file" ]]; then
    produced=$(sed -n '2p' "$out_file" | tr -d '\r\n')
fi
if [[ -f "$exp_file" ]]; then
    expected=$(sed -n '2p' "$exp_file" | tr -d '\r\n')
fi

match="NO"
[[ "$produced" == "$expected" && "$produced" != "" && "$produced" != "NA" ]] && match="YES"

printf "case01: time=%0.2fs produced=%s expected=%s match=%s\n" "$elapsed" "$produced" "$expected" "$match"

