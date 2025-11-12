#!/usr/bin/env bash
set -euo pipefail

BIN=./hw4
TEST_DIR=testcases
OUT_DIR=outputs
LOG_DIR=logs
LOG_FILE=$LOG_DIR/test_results.log

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Temporary log for current run
TMP_LOG=$(mktemp)

{
    echo "==== Test Run $(date) ===="
    echo "Case,Elapsed(s),NonceProduced,NonceExpected,Match"
} > "$TMP_LOG"

total_time=0

for i in 0 1 2 3; do
    in_file="$TEST_DIR/case0$i.in"
    exp_file="$TEST_DIR/case0$i.out"
    out_file="$OUT_DIR/run_case0$i.out"

    if [[ ! -f "$in_file" ]]; then
        echo "Skipping case0$i (missing $in_file)"
        continue
    fi
    if [[ ! -x "$BIN" ]]; then
        echo "Binary $BIN not executable."
        exit 1
    fi

    start_time=$(date +%s.%N)
    "$BIN" "$in_file" "$out_file" >/dev/null 2>&1 || echo "Execution error case0$i"
    end_time=$(date +%s.%N)

    # Compute elapsed time to 2 decimal places
    elapsed=$(echo "$end_time - $start_time" | bc)
    elapsed=$(printf "%.2f" "$elapsed")

    # Accumulate total runtime
    total_time=$(echo "$total_time + $elapsed" | bc)

    # Extract produced/expected nonce (2nd line)
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

    printf "case0%d: time=%0.2fs produced=%s expected=%s match=%s\n" "$i" "$elapsed" "$produced" "$expected" "$match"
    printf "case0%d,%0.2f,%s,%s,%s\n" "$i" "$elapsed" "$produced" "$expected" "$match" >> "$TMP_LOG"
done

# Determine if this run is the best (minimum total runtime among all logs)
is_best="NO"
if [[ -f "$LOG_FILE" ]]; then
    # Extract previous total runtimes (from lines starting with "Total,")
    prev_best=$(grep "^Total," "$LOG_FILE" | awk -F',' '{print $2}' | sort -n | head -n 1)
    # Compare with current total_time
    if [[ -z "$prev_best" || $(echo "$total_time < $prev_best" | bc) -eq 1 ]]; then
        is_best="YES"
    fi
else
    is_best="YES"
fi

# Print summary line and append to tmp log
printf "Total runtime: %.2fs (%s)\n" "$total_time" "$is_best"
printf "Total,%.2f,%s\n" "$total_time" "$is_best" >> "$TMP_LOG"

# Prepend new log block to existing log
if [[ -f "$LOG_FILE" ]]; then
    cat "$LOG_FILE" >> "$TMP_LOG"
fi
mv "$TMP_LOG" "$LOG_FILE"

echo "Log updated (latest run on top): $LOG_FILE"

