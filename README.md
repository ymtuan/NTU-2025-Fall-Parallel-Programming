# 2025-Fall-Parallel-Programming-hw5


- Sequential P1/P2 on separate GPUs - Avoids resource contention
- Single GPU per problem - 2-GPU split has too much overhead for this n
- Removed checkpoint overhead - Not needed, slows down P2
- Optimized memory copies - Batch copies together
- Removed unnecessary syncs - Only sync at end of P3
- Better GPU utilization - P1 on GPU0, P2 on GPU1 sequentially, then P3 uses both