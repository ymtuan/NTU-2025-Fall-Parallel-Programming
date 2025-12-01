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

## 183.14
- Checkpointing (Major Speedup): During the main simulation (Problem 2), the system saves a "snapshot" of the universe every 1,000 steps. Problem 3 tasks now load the nearest checkpoint instead of restarting from step 0, reducing computation per task by >99%.
- Parallel Multi-GPU Workflow: Problem 1 (GPU 1) and Problem 2 (GPU 0) run simultaneously on separate threads. The final Problem 3 tasks are distributed across both GPUs.
- Kernel Optimizations: Implemented Shared Memory Tiling to optimize the $O(N^2)$ memory bandwidth, stored parameters in Constant Memory, and used Double Buffering to pipeline updates without race conditions.
- Pre-calculation: Missile hit times are calculated once during the main simulation, so Problem 3 only simulates valid destruction scenarios.

  b40    5.84   accepted  
  b20    5.19   accepted  
  b30    4.78   accepted  
  b70    6.59   accepted  
  b50    6.99   accepted  
  b60    7.04   accepted  
  b80    7.64   accepted  
 b100    8.04   accepted  
  b90   11.15   accepted  
 b200   24.82   accepted  
 b512   34.74   accepted  
b1024   60.33   accepted

## 164.01
- Elimination of Control Flow (Branchless Loop)

  - What we did: We removed the if (idx >= n) break; check inside the innermost calculation loop.

  - How: We "padded" the Shared Memory tiles with Ghost Particles (particles with mass = 0) for any index outside the valid range.

  - Why it helps: The GPU hates conditional jumps inside tight loops. By ensuring every thread runs exactly 256 iterations, the compiler can fully pipeline instructions and utilize Loop Unrolling, leading to much higher arithmetic throughput (FLOPs).

- Shared Memory Caching for Collision Checks

  - What we did: We loaded the Planet's position into __shared__ memory at the very start of the kernel (using just Thread 0).

  - Why it helps: During the "Missile Check" phase, every single thread needs to calculate the distance to the planet. Previously, this caused massive contention on Global Memory. Now, all threads read from the fast L1/Shared cache.

- Instruction Level Parallelism (ILP)

  - What we did: By combining the fixed loop count with #pragma unroll, we allowed the GPU to issue multiple Multiply-Add (FMA) instructions simultaneously, hiding the latency of memory fetches.


  b40    3.13   accepted  
  b20    2.03   accepted  
  b30    2.33   accepted  
  b70    4.03   accepted  
  b50    3.98   accepted  
  b60    4.43   accepted  
  b90    4.49   accepted  
  b80    4.58   accepted  
 b100    5.44   accepted  
 b512   35.13   accepted  
 b200   14.55   accepted  
b1024   79.90   accepted