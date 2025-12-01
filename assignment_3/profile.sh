#!/bin/bash

echo "=== CUDA Performance Profiling for Mandelbulb ==="
echo ""

# Test case parameters
EXECUTABLE="./hw3"
TEST_CASE="testcases/00.txt"

# Extract parameters
PARAMS=$(awk '
/pos=/ { 
    split($0, a, "="); 
    split(a[2], b, " "); 
    px=b[1]; py=b[2]; pz=b[3]
}
/tarpos=/ { 
    split($0, a, "="); 
    split(a[2], b, " "); 
    tx=b[1]; ty=b[2]; tz=b[3]
}
/width=/ { 
    split($0, a, "="); 
    w=a[2]
}
/height=/ { 
    split($0, a, "="); 
    h=a[2]
}
END { print px, py, pz, tx, ty, tz, w, h }
' "$TEST_CASE")

echo "Test Parameters: $PARAMS"
echo ""

# Profile with nvprof
echo "=== Running nvprof for kernel profiling ==="
nvprof --print-gpu-trace $EXECUTABLE $PARAMS output_profile.png 2>&1 | tail -50
echo ""

# Profile with ncu (Nsight Compute)
echo "=== Running ncu (Nsight Compute) ==="
echo "This will generate detailed metrics..."
ncu --set full --export profile.ncu-rep $EXECUTABLE $PARAMS output_profile.png

echo ""
echo "Profile report saved to: profile.ncu-rep"
echo "To view: ncu --import profile.ncu-rep"
