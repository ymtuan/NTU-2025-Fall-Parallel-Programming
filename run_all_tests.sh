#!/bin/bash

# ==============================================================================
# Script to run all SIFT test cases, validate the results, and log performance.
#
# Usage:
#   1. Make sure your 'hw2' executable is compiled ('make').
#   2. Configure MPI/OpenMP parameters below.
#   3. Run from your hw2 directory: ./run_all_tests.sh
#   4. View the summary in the terminal and details in 'runtime_log.txt'.
# ==============================================================================

# --- SLURM & MPI Configuration ---
ACCOUNT="ACD114118"
NODES=2
PROCS=4
CORES=6
TIME_LIMIT="00:03:00"

# --- Log File & Directories ---
LOG_FILE="runtime_log.txt"
RESULTS_DIR="results"
GOLDENS_DIR="goldens"
TESTCASES_DIR="testcases"

# --- Color Codes for Terminal Output ---
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# --- Pre-run Checks ---
if [ ! -f "hw2" ]; then
    echo "${RED}Error: Executable 'hw2' not found. Please compile with 'make'.${RESET}"
    exit 1
fi
if [ ! -f "validate.py" ]; then
    echo "${RED}Error: 'validate.py' not found in the current directory.${RESET}"
    exit 1
fi
if [ ! -d "$GOLDENS_DIR" ]; then
    echo "${RED}Error: Golden results directory '$GOLDENS_DIR' not found.${RESET}"
    exit 1
fi

# Create results directory if it doesn't exist.
mkdir -p "$RESULTS_DIR"

# --- Script Execution ---
echo "SIFT Execution & Validation Log - $(date)" > "$LOG_FILE"
echo "Running with: ${NODES} Nodes, ${PROCS} Processes, ${CORES} Cores per Process" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"
echo ""

echo "Starting tests... (Detailed output will be in $LOG_FILE)"
echo "--------------------------------------------------------"

# Loop through all 8 test cases.
for i in {1..3}
do
    TEST_CASE=$(printf "%02d" "$i")

    INPUT_IMG="${TESTCASES_DIR}/${TEST_CASE}.jpg"
    OUTPUT_IMG="${RESULTS_DIR}/${TEST_CASE}.jpg"
    OUTPUT_TXT="${RESULTS_DIR}/${TEST_CASE}.txt"
    GOLDEN_IMG="${GOLDENS_DIR}/${TEST_CASE}.jpg"
    GOLDEN_TXT="${GOLDENS_DIR}/${TEST_CASE}.txt"

    # Print status to the command line.
    printf "Running Test Case %s... " "${TEST_CASE}"

    # Construct the srun command and execute it, capturing all output.
    SRUN_CMD="srun -A ${ACCOUNT} -N ${NODES} -n ${PROCS} -c ${CORES} --time=${TIME_LIMIT} ./hw2 ${INPUT_IMG} ${OUTPUT_IMG} ${OUTPUT_TXT}"
    EXECUTION_OUTPUT=$(eval "$SRUN_CMD" 2>&1)

    # Append the detailed execution output to the log file.
    echo "--- Test Case ${TEST_CASE} ---" >> "$LOG_FILE"
    echo "Command: $SRUN_CMD" >> "$LOG_FILE"
    echo "$EXECUTION_OUTPUT" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Check if the output text file was created. If not, the run failed.
    if [ ! -f "$OUTPUT_TXT" ]; then
        printf "${RED}[EXECUTION FAILED]${RESET}\n"
        echo "Validation: Skipped because output file was not created." >> "$LOG_FILE"
        continue
    fi
    
    # Run the validation script and capture its simple "Pass" or "Wrong" output.
    VALIDATION_RESULT=$(python3 validate.py "$OUTPUT_TXT" "$GOLDEN_TXT" "$OUTPUT_IMG" "$GOLDEN_IMG")
    echo "Validation Result: ${VALIDATION_RESULT}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Extract the execution time from the captured output.
    RUNTIME=$(echo "$EXECUTION_OUTPUT" | grep "Execution time" | sed 's/Execution time: //')

    # Display the final summary for this test case.
    if [[ "$VALIDATION_RESULT" == "Pass" ]]; then
        printf "${GREEN}[PASS]${RESET}"
    else
        printf "${RED}[WRONG]${RESET}"
    fi
    
    printf " - Runtime: ${YELLOW}%s${RESET}\n" "${RUNTIME:-'Not found'}"

done

echo "--------------------------------------------------------"
echo "All tests are complete! Check '$LOG_FILE' for details."

