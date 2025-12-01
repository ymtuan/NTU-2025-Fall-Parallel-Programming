#!/bin/bash
trap "echo 'Interrupted!'; exit" SIGINT

for i in $(seq -w 1 25); do
    echo "Running sample $i ..."
    start=$(date +%s.%N)
    ./hw1 "samples/${i}.txt" > output.txt 2>&1
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    python3 validate.py samples/${i}.txt output.txt
    echo "Elapsed time: $elapsed seconds"
    echo ""
done
