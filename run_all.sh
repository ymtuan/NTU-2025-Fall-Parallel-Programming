#!/bin/bash

# --- Configuration ---
EXECUTABLE="./hw3"           # Path to your compiled CUDA executable
TESTCASE_DIR="testcases"     # Directory containing test case files (xx.txt and xx.png)
LOG_FILE="log.txt"           # File to store runtime and accuracy results
COMPARATOR="./compare_png"   # Path to the compiled C++ comparison tool
COMPARATOR_SRC="compare_png.cpp" # Source file for the comparison tool
LODEPNG_SRC="lodepng/lodepng.cpp" # Source file for lodepng library

# --- Ensure Directories and Files Exist ---
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found."
    echo "Please compile your hw3.cu first (e.g., run 'make')."
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

# --- Compile Comparator ---
if [ "$COMPARATOR_SRC" -nt "$COMPARATOR" ] || [ ! -f "$COMPARATOR" ] || [ "$LODEPNG_SRC" -nt "$COMPARATOR" ]; then
    echo "Compiling PNG comparator tool..."
    g++ "$COMPARATOR_SRC" "$LODEPNG_SRC" -o "$COMPARATOR" -O3 -Wall || {
        echo "Error: Failed to compile comparator tool '$COMPARATOR_SRC'."
        exit 1
    }
    echo "Comparator compiled successfully."
fi

# --- Run Tests ---
> "$LOG_FILE"

echo "Starting Mandelbulb Tests..." | tee -a "$LOG_FILE"
printf "| %-10s | %-15s | %-10s |\n" "Test Case" "Runtime (s)" "Accuracy (%)" | tee -a "$LOG_FILE"
echo "|------------|-----------------|------------|" | tee -a "$LOG_FILE"

# Loop through test cases 00 to 08
for i in $(seq -f "%02g" 0 8); do
    TEST_PARAM_FILE="${TESTCASE_DIR}/${i}.txt"
    GROUND_TRUTH_PNG="${TESTCASE_DIR}/${i}.png"
    OUTPUT_PNG="output_${i}.png"

    # --- Check if input files exist ---
    if [ ! -f "$TEST_PARAM_FILE" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Param missing" "N/A" | tee -a "$LOG_FILE"
        echo "  Skipping Test Case ${i}: Parameter file '$TEST_PARAM_FILE' not found." >&2
        continue
    fi
    
    if [ ! -f "$GROUND_TRUTH_PNG" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "GT missing" "N/A" | tee -a "$LOG_FILE"
        echo "  Skipping Test Case ${i}: Ground truth file '$GROUND_TRUTH_PNG' not found." >&2
        continue
    fi

    # --- Extract parameters ---
    # Parse the format: pos=x y z, tarpos=x y z, width=w, height=h
    px=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $1}')
    py=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $2}')
    pz=$(grep "^pos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $3}')
    
    tx=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $1}')
    ty=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $2}')
    tz=$(grep "^tarpos=" "$TEST_PARAM_FILE" | cut -d'=' -f2 | awk '{print $3}')
    
    w=$(grep "^width=" "$TEST_PARAM_FILE" | cut -d'=' -f2)
    h=$(grep "^height=" "$TEST_PARAM_FILE" | cut -d'=' -f2)

    # Check if parameters were extracted
    if [ -z "$px" ] || [ -z "$py" ] || [ -z "$pz" ] || [ -z "$tx" ] || [ -z "$ty" ] || [ -z "$tz" ] || [ -z "$w" ] || [ -z "$h" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Parse error" "N/A" | tee -a "$LOG_FILE"
        echo "  Skipping Test Case ${i}: Failed to parse parameters from '$TEST_PARAM_FILE'." >&2
        echo "  Extracted: px=$px py=$py pz=$pz tx=$tx ty=$ty tz=$tz w=$w h=$h" >&2
        continue
    fi

    PARAMS="$px $py $pz $tx $ty $tz $w $h"
    echo "  Test Case ${i}: Parameters: $PARAMS" >&2

    # --- Execute and Time ---
    TIMEOUT_DURATION=600
    START_TIME=$(date +%s.%N)
    timeout $TIMEOUT_DURATION $EXECUTABLE $PARAMS "$OUTPUT_PNG" > /dev/null 2>&1
    EXECUTION_EXIT_CODE=$?
    END_TIME=$(date +%s.%N)
    
    # Calculate runtime
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    # Check execution status
    if [ $EXECUTION_EXIT_CODE -eq 124 ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Timeout" "N/A" | tee -a "$LOG_FILE"
        echo "  Test Case ${i}: Timed out after ${TIMEOUT_DURATION} seconds." >&2
        continue
    elif [ $EXECUTION_EXIT_CODE -ne 0 ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "Exec Error" "N/A" | tee -a "$LOG_FILE"
        echo "  Test Case ${i}: Execution failed with exit code ${EXECUTION_EXIT_CODE}." >&2
        continue
    fi

    # --- Compare Images ---
    if [ ! -f "$OUTPUT_PNG" ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "No Output" | tee -a "$LOG_FILE"
        echo "  Test Case ${i}: Output file '$OUTPUT_PNG' was not generated." >&2
        continue
    fi

    ACCURACY=$($COMPARATOR "$OUTPUT_PNG" "$GROUND_TRUTH_PNG")
    COMPARE_EXIT_CODE=$?

    if [ $COMPARE_EXIT_CODE -ne 0 ]; then
        printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "Comp Error" | tee -a "$LOG_FILE"
        echo "  Test Case ${i}: Comparison tool failed." >&2
    else
        printf "| %-10s | %-15s | %-10s |\n" "$i" "$RUNTIME" "$ACCURACY" | tee -a "$LOG_FILE"
    fi

done

echo "|------------|-----------------|------------|" | tee -a "$LOG_FILE"
echo "Testing finished. Results logged to $LOG_FILE"
