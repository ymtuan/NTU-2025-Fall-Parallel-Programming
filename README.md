# 2025-Fall-Parallel-Programming-hw5

## 401.17
- Sequential P1/P2 on separate GPUs - Avoids resource contention
- Single GPU per problem - 2-GPU split has too much overhead for this n
- Removed checkpoint overhead - Not needed, slows down P2
- Optimized memory copies - Batch copies together
- Removed unnecessary syncs - Only sync at end of P3
- Better GPU utilization - P1 on GPU0, P2 on GPU1 sequentially, then P3 uses both

## 272.30
- Host-Side Parallelism (P1 & P2)In main, Problem 1 (on GPU 0) and Problem 2 (on GPU 1) are now launched in separate std::threads and run concurrently, significantly reducing the total execution time before starting Problem 3.
- Kernel-Based Data Broadcast (P3)
A new kernel, broadcast_kernel, is introduced to drastically reduce host-to-device (H2D) memory transfers in run_batch_sim.
    - The host now performs only one H2D copy of the initial state (1xN) into the first segment of each batched device array (d_qx, d_qy, etc.).
    - The broadcast_kernel is then launched to replicate this initial state from the first segment to all other $(N_{\text{batch}}-1)$ segments directly on the GPU, which is much faster than multiple H2D transfers.