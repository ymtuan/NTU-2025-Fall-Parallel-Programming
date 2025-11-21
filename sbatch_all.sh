#!/bin/bash
#SBATCH --job-name=hw5_test
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o hw5_all.out
#SBATCH -e hw5_all.err

# Testcases
testcases=(20 30 40 50 60 70 80 90 100 200 512 1024)

SUMMARY_TXT="summary.txt"
SUMMARY_CSV="summary.csv"

echo "HW5 Test Summary" > $SUMMARY_TXT
echo "testcase,runtime_sec,status" > $SUMMARY_CSV

for tc in "${testcases[@]}"; do
    echo "===== Running testcase b$tc ====="

    # High-precision start time (seconds.milliseconds)
    start=$(date +%s.%N)

    # Execute with timeout 180s
    timeout 200s ./hw5 ../testcases/b${tc}.in my_output_${tc}.out
    status_hw5=$?

    end=$(date +%s.%N)

    # Compute runtime to 2 decimal places using bc
    runtime=$(echo "$end - $start" | bc)
    runtime2=$(printf "%.2f" "$runtime")

    # Check timeout or crash
    if [ $status_hw5 -eq 124 ]; then
        echo "b$tc TIMEOUT (200s)" | tee -a $SUMMARY_TXT
        echo "$tc,200.00,TIMEOUT" >> $SUMMARY_CSV
        continue
    elif [ $status_hw5 -ne 0 ]; then
        echo "b$tc CRASH (exit $status_hw5)" | tee -a $SUMMARY_TXT
        echo "$tc,$runtime2,CRASH" >> $SUMMARY_CSV
        continue
    fi

    # Run validator and capture its output
    validate_output=$(python3 ../validate.py my_output_${tc}.out ../testcases/b${tc}.out)

    if [ "$validate_output" = "ok" ]; then
        echo "b$tc OK ($runtime2 s)" | tee -a $SUMMARY_TXT
        echo "$tc,$runtime2,OK" >> $SUMMARY_CSV
    else
        echo "b$tc WRONG ($runtime2 s)" | tee -a $SUMMARY_TXT
        echo "  -> Wrong fields: $validate_output" | tee -a $SUMMARY_TXT
        echo "$tc,$runtime2,WRONG" >> $SUMMARY_CSV
    fi

    echo
done

echo "All testcases finished." >> $SUMMARY_TXT
