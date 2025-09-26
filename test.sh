#!/bin/bash
trap "echo 'Interrupted!'; exit" SIGINT

for i in $(seq -w 1 25); do
    echo "Running sample $i ..."
    #start=$(date +%s.%N)
    ./hw1 "samples/${i}.txt"
    #end=$(date +%s.%N)
    #elapsed=$(echo "$end - $start" | bc)
    #echo "Elapsed time: $elapsed seconds"
    echo ""
done
