#!/bin/bash

# --- Configuration ---
EXECUTABLE="./hw3"
TESTCASE_DIR="testcases"
LOG_FILE="log.txt"
COMPARATOR="./compare_png"
COMPARATOR_SRC="compare_png.cpp"
LODEPNG_SRC="lodepng/lodepng.cpp"

# --- Ensure Directories and Files Exist ---
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found."
    exit 1
fi

if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: Testcase directory '$TESTCASE_DIR' not found."
    exit 1
fi

if [ ! -f "$COMPARATOR_SRC" ]; then
    echo "Error: Comparator source file '$COMPARATOR_SRC' not found."
    exit 1
fi

if [ ! -f "$LODEPNG_SRC" ]; then
    echo "Error: Lodepng source file '$LODEPNG_SRC' not found."
    exit 1
fi

# --- Compile Comparator (only when needed) ---
if [ "$COMPARATOR_SRC" -nt "$COMPARATOR" ] || [ ! -f "$COMPARATOR" ] || [ "$LODEPNG_SRC" -nt "$COMPARATOR" ]; then
    echo "Compiling PNG comparator tool..."
    g++ "$COMPARATOR_SRC" "$LODEPNG_SRC" -o "$COMPARATOR" -O3 -Wall || {
        echo "Error: Failed to compile comparator tool."
        exit 1
    }
fi

# --- Run Tests ---
> "$LOG_FILE"
TOTAL_TIME=0
ALL_PASS=true

echo "Starting Mandelbulb Tests..." | tee -a "$LOG_FILE"
printf "| %-10s | %-15s | %-10s |\n" "Test Case" "Runtime (s)" "Accuracy (%)" | tee -a "$LOG_FILE"
echo "|------------|-----------------|------------|" | tee -a "$LOG_FILE"

for i in $(seq -f "%02g" 0 8); do
    TEST_PARAM_FILE="${TESTCASE_DIR}/${i}.txt"
    GROUND_TRUTH_PNG="${TESTCASE_DIR}/${i}.png"
    OUTPUT_PNG="output_${i}.png"

    if [ ! -f "$TEST_PARAM_FILE" ] || [ ! -f "$GROUND_TRUTH_PNG" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "File missing" "N/A" | tee -a "$LOG_FILE"
        continue
    fi

    px=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $1}')
    py=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $2}')
    pz=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $3}')
    tx=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $1}')
    ty=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $2}')
    tz=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $3}')
    w=$(grep "^width=" "$TEST_PARAM_FILE" | cut -d'=' -f2)
    h=$(grep "^height=" "$TEST_PARAM_FILE" | cut -d'=' -f2)

    if [ -z "$px" ] || [ -z "$py" ] || [ -z "$pz" ] || [ -z "$tx" ] || [ -z "$ty" ] || [ -z "$tz" ] || [ -z "$w" ] || [ -z "$h" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Parse error" "N/A" | tee -a "$LOG_FILE"
        continue
    fi

    TIMEOUT_DURATION=600
    START_TIME=$(date +%s.%N)
    timeout $TIMEOUT_DURATION $EXECUTABLE "$px" "$py" "$pz" "$tx" "$ty" "$tz" "$w" "$h" "$OUTPUT_PNG" > /dev/null 2>&1
    EXIT_CODE=$?
    END_TIME=$(date +%s.%N)
    
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    if [ $EXIT_CODE -ne 0 ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Exec error" "N/A" | tee -a "$LOG_FILE"
        ALL_PASS=false
        continue
    fi

    if [ ! -f "$OUTPUT_PNG" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "No Output" | tee -a "$LOG_FILE"
        ALL_PASS=false
        continue
    fi

    ACCURACY=$($COMPARATOR "$OUTPUT_PNG" "$GROUND_TRUTH_PNG")
    if [ $? -ne 0 ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "Comp Err" | tee -a "$LOG_FILE"
        ALL_PASS=false
        continue
    fi

    printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "$ACCURACY" | tee -a "$LOG_FILE"

    TOTAL_TIME=$(echo "$TOTAL_TIME + $RUNTIME" | bc)

    # --- Check accuracy threshold ---
    ACC_VAL=$(printf "%.0f" "$ACCURACY")
    if [ "$ACC_VAL" -lt 97 ]; then
        ALL_PASS=false
    fi
done

echo "|------------|-----------------|------------|" | tee -a "$LOG_FILE"

# --- Summary ---
echo "Total Runtime: ${TOTAL_TIME}s" | tee -a "$LOG_FILE"

if $ALL_PASS; then
    echo "All test cases passed with accuracy ≥ 97%" | tee -a "$LOG_FILE"
else
    echo "Some test cases had accuracy below 97% or errors occurred." | tee -a "$LOG_FILE"
fi

echo "Testing finished. Results logged to $LOG_FILE"
