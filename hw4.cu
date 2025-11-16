#include <cstdio>
#include <cstring>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "sha256.h"

// ------------------ Structures ------------------
typedef struct _block {
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

// ------------------ CUDA Error Helper ------------------
#undef CUDA_OK
#undef CUDA_OK_RET
#define CUDA_CALL_OR_FALSE(stmt) do { cudaError_t _e = (stmt); if(_e != cudaSuccess){ \
  fprintf(stderr,"CUDA Error %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_e), _e); return false; } } while(0)

// ------------------ Hex Decode ------------------
__host__ __device__ unsigned char decode(unsigned char c) {
    switch(c) {
        case '0'...'9': return c - '0';
        case 'a': case 'A': return 0x0a;
        case 'b': case 'B': return 0x0b;
        case 'c': case 'C': return 0x0c;
        case 'd': case 'D': return 0x0d;
        case 'e': case 'E': return 0x0e;
        case 'f': case 'F': return 0x0f;
        default: return 0;
    }
}

// ------------------ Endian Conversion ------------------
void convert_string_to_little_endian_bytes(unsigned char* out, const char *in, size_t hex_len) {
    assert(hex_len % 2 == 0);
    size_t s = 0;
    size_t b = hex_len/2 - 1;
    for(; s < hex_len; s += 2, --b) {
        out[b] = (unsigned char)((decode(in[s]) << 4) | decode(in[s+1]));
    }
}

// ------------------ Printing Helpers ------------------
void print_hex(const unsigned char* hex, size_t len) {
    for(size_t i=0;i<len;++i) printf("%02x", hex[i]);
}
void print_hex_inverse(const unsigned char* hex, size_t len) {
    for(int i=(int)len-1;i>=0;--i) printf("%02x", hex[i]);
}

// ------------------ Comparison (little-endian arrays) ------------------
__host__ __device__ int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len) {
    for(int i = (int)byte_len - 1; i >= 0; --i) {
        if(a[i] < b[i]) return -1;
        if(a[i] > b[i]) return 1;
    }
    return 0;
}

// ------------------ Safe getline (file) ------------------
void getline_s(char *str, size_t len, FILE *fp) {
    int i=0;
    int ch;
    while(i < (int)len-1 && (ch = fgetc(fp)) != EOF && ch != '\n') {
        str[i++] = (char)ch;
    }
    str[i] = '\0';
}

static inline bool read_line(FILE *fp, char *buf, size_t cap) {
    if(!fgets(buf, (int)cap, fp)) return false;
    size_t n = strlen(buf);
    while(n && (buf[n-1]=='\n' || buf[n-1]=='\r')) buf[--n] = 0;
    return true;
}

// ------------------ double SHA256 (host + device) ------------------
__host__ __device__ inline void double_sha256(SHA256 *out_ctx, const unsigned char *bytes, size_t len) {
    SHA256 tmp;
    sha256(&tmp, bytes, len);
    sha256(out_ctx, (const unsigned char*)&tmp, sizeof(tmp));
}

// ------------------ Merkle Root (host) ------------------
void calc_merkle_root(unsigned char *root, int count, char **branch) {
    size_t total = count;
    if(total == 0) { memset(root, 0, 32); return; }
    unsigned char *raw = new unsigned char[(total+1)*32];
    unsigned char **list = new unsigned char*[total+1];

    for(int i=0;i<count;++i) {
        list[i] = raw + i*32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }
    list[total] = raw + total*32;

    while(total > 1) {
        if(total & 1) memcpy(list[total], list[total-1], 32);
        size_t write = 0;
        for(size_t i=0;i<total;i+=2, ++write) {
            double_sha256((SHA256*)list[write], list[i], 64);
        }
        total = write;
    }
    memcpy(root, list[0], 32);
    delete[] list;
    delete[] raw;
}

// ------------------ Device Globals ------------------
__constant__ unsigned char d_target[32];
__constant__ WORD d_midstate[8];  // Pre-computed midstate for first 64 bytes
__device__ unsigned int d_solution_nonce;
__device__ int d_found;
__device__ unsigned char d_solution_hash[32];

// ------------------ Optimized Mining Kernel with Midstate ------------------
// __global__ void mine_kernel_midstate(const unsigned char *last12bytes, unsigned long long max_nonce_space) {
//     unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
//     unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;

//     // Load last12bytes into registers once (avoid repeated global memory access)
//     unsigned char l0=last12bytes[0], l1=last12bytes[1], l2=last12bytes[2], l3=last12bytes[3];
//     unsigned char l4=last12bytes[4], l5=last12bytes[5], l6=last12bytes[6], l7=last12bytes[7];
//     unsigned char l8=last12bytes[8], l9=last12bytes[9], l10=last12bytes[10], l11=last12bytes[11];

//     // Batch atomic checks - only check every 256 iterations to reduce contention
//     int local_batch = 0;

//     for(unsigned long long nonce = tid; nonce < max_nonce_space; nonce += total_threads) {
//         // Batched early exit check (every 256 iterations)
//         if((local_batch & 0xFF) == 0 && d_found) return;
//         local_batch++;

//         // Build last 16 bytes directly using register values
//         unsigned char last16[16];
//         last16[0]=l0; last16[1]=l1; last16[2]=l2; last16[3]=l3;
//         last16[4]=l4; last16[5]=l5; last16[6]=l6; last16[7]=l7;
//         last16[8]=l8; last16[9]=l9; last16[10]=l10; last16[11]=l11;

//         // Nonce (little-endian, 4 bytes)
//         last16[12] = (unsigned char)(nonce);
//         last16[13] = (unsigned char)(nonce >> 8);
//         last16[14] = (unsigned char)(nonce >> 16);
//         last16[15] = (unsigned char)(nonce >> 24);

//         // First SHA256: finalize from midstate
//         SHA256 ctx1;
//         sha256_finalize_from_midstate(&ctx1, d_midstate, last16);

//         // Second SHA256: optimized for 32-byte input
//         SHA256 ctx2;
//         sha256_32bytes(&ctx2, ctx1.b);

//         if(little_endian_bit_comparison(ctx2.b, d_target, 32) < 0) {
//             // Only update if we're first
//             if(atomicCAS(&d_found, 0, 1) == 0) {
//                 d_solution_nonce = (unsigned int)nonce;
//                 for(int i=0;i<32;++i) d_solution_hash[i] = ctx2.b[i];
//             }
//             return;
//         }
//     }
// }

// 1. Use constant memory for last12 (definite win, no downside)
__constant__ unsigned char d_last12[12];

// 2. Original kernel structure but with constant memory load
__global__ void mine_kernel_midstate(unsigned long long max_nonce_space) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;

    // Load from constant memory (cached, broadcast)
    unsigned char l0=d_last12[0], l1=d_last12[1], l2=d_last12[2], l3=d_last12[3];
    unsigned char l4=d_last12[4], l5=d_last12[5], l6=d_last12[6], l7=d_last12[7];
    unsigned char l8=d_last12[8], l9=d_last12[9], l10=d_last12[10], l11=d_last12[11];

    int local_batch = 0;

    for(unsigned long long nonce = tid; nonce < max_nonce_space; nonce += total_threads) {
        // Reduced frequency check (every 512 instead of 256)
        if((local_batch & 0x1FF) == 0 && d_found) return;
        local_batch++;

        // Keep original structure - don't try to move array outside
        unsigned char last16[16];
        last16[0]=l0; last16[1]=l1; last16[2]=l2; last16[3]=l3;
        last16[4]=l4; last16[5]=l5; last16[6]=l6; last16[7]=l7;
        last16[8]=l8; last16[9]=l9; last16[10]=l10; last16[11]=l11;
        last16[12] = (unsigned char)(nonce);
        last16[13] = (unsigned char)(nonce >> 8);
        last16[14] = (unsigned char)(nonce >> 16);
        last16[15] = (unsigned char)(nonce >> 24);

        SHA256 ctx1;
        sha256_finalize_from_midstate_opt(&ctx1, d_midstate, last16);

        SHA256 ctx2;
        sha256_32bytes_opt(&ctx2, ctx1.b);

        if(little_endian_bit_comparison(ctx2.b, d_target, 32) < 0) {
            if(atomicCAS(&d_found, 0, 1) == 0) {
                d_solution_nonce = (unsigned int)nonce;
                for(int i=0;i<32;++i) d_solution_hash[i] = ctx2.b[i];
            }
            return;
        }
    }
}

// ------------------ Target Computation ------------------
void compute_target(unsigned char target_hex[32], unsigned int nbits_le) {
    unsigned int exp  = nbits_le >> 24;
    unsigned int mant = nbits_le & 0x00ffffff;
    memset(target_hex, 0, 32);
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    // little-endian
    if (sb < 32) target_hex[sb    ] = (unsigned char)(mant << rb);
    if (sb + 1 < 32) target_hex[sb + 1] = (unsigned char)(mant >> (8 - rb));
    if (sb + 2 < 32) target_hex[sb + 2] = (unsigned char)(mant >> (16 - rb));
    if (sb + 3 < 32) target_hex[sb + 3] = (unsigned char)(mant >> (24 - rb));
}

// ------------------ GPU Miner ------------------
// bool gpu_mine(HashBlock &block, const unsigned char target_hex[32],
//               unsigned int &out_nonce, unsigned char out_hash[32],
//               unsigned long long max_nonce_space) {

//     // Compute midstate for first 64 bytes
//     unsigned char first64[64];
//     memcpy(first64, &block, 64);
//     WORD midstate[8];
//     sha256_compute_midstate(first64, midstate);

//     // Upload midstate and target to device
//     CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_midstate, midstate, sizeof(midstate)));
//     CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_target, target_hex, 32));

//     // Extract last 12 bytes (before nonce) - bytes 64-75
//     unsigned char last12[12];
//     memcpy(last12, ((unsigned char*)&block) + 64, 12);

//     // Upload last12 bytes to device
//     unsigned char *d_last12;
//     CUDA_CALL_OR_FALSE(cudaMalloc(&d_last12, 12));
//     CUDA_CALL_OR_FALSE(cudaMemcpy(d_last12, last12, 12, cudaMemcpyHostToDevice));

//     int zero = 0;
//     CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_found, &zero, sizeof(int)));
//     unsigned int init_nonce = 0;
//     CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_solution_nonce, &init_nonce, sizeof(unsigned int)));

//     // Increased parallelism - more threads = better GPU utilization
//     int threads = 512;
//     int blocks = 65535;

//     mine_kernel_midstate<<<blocks, threads>>>(d_last12, max_nonce_space);
//     cudaError_t err = cudaDeviceSynchronize();
//     cudaFree(d_last12);

//     if(err != cudaSuccess) {
//         fprintf(stderr,"Kernel failure: %s\n", cudaGetErrorString(err));
//         return false;
//     }

//     int h_found = 0;
//     CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));
//     if(!h_found) return false;

//     CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(&out_nonce, d_solution_nonce, sizeof(unsigned int)));
//     CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(out_hash, d_solution_hash, 32));
//     return true;
// }

// Updated gpu_mine - just change last12 to constant memory
bool gpu_mine(HashBlock &block, const unsigned char target_hex[32],
              unsigned int &out_nonce, unsigned char out_hash[32],
              unsigned long long max_nonce_space) {

    unsigned char first64[64];
    memcpy(first64, &block, 64);
    WORD midstate[8];
    sha256_compute_midstate(first64, midstate);

    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_midstate, midstate, sizeof(midstate)));
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_target, target_hex, 32));

    unsigned char last12[12];
    memcpy(last12, ((unsigned char*)&block) + 64, 12);
    
    // Upload to constant memory instead of global malloc
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_last12, last12, 12));

    int zero = 0;
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_found, &zero, sizeof(int)));
    unsigned int init_nonce = 0;
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_solution_nonce, &init_nonce, sizeof(unsigned int)));

    int threads = 512;
    int blocks = 65535;

    // No more d_last12 pointer parameter!
    mine_kernel_midstate<<<blocks, threads>>>(max_nonce_space);
    
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        fprintf(stderr,"Kernel failure: %s\n", cudaGetErrorString(err));
        return false;
    }

    int h_found = 0;
    CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(&h_found, d_found, sizeof(int)));
    if(!h_found) return false;

    CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(&out_nonce, d_solution_nonce, sizeof(unsigned int)));
    CUDA_CALL_OR_FALSE(cudaMemcpyFromSymbol(out_hash, d_solution_hash, 32));
    return true;
}

// ------------------ Solve One Block ------------------
void solve(FILE *fin, FILE *fout) {
    // Parse exactly like sequential version (tokens, not whole raw lines)
    char version[9] = {0};
    char prevhash[65] = {0};
    char ntime[9] = {0};
    char nbits[9] = {0};
    int tx = 0;

    if (fscanf(fin, "%8s", version) != 1 ||
        fscanf(fin, "%64s", prevhash) != 1 ||
        fscanf(fin, "%8s", ntime) != 1 ||
        fscanf(fin, "%8s", nbits) != 1 ||
        fscanf(fin, "%d", &tx) != 1)
    {
        fprintf(stderr, "Parse failure (header fields)\n");
        fprintf(fout, "ffffffff\n");
        return;
    }

    if (tx <= 0) {
        fprintf(stderr, "Invalid tx count %d\n", tx);
        fprintf(fout, "ffffffff\n");
        return;
    }

    // Consume trailing newline after tx (if any)
    int c;
    while ((c = fgetc(fin)) == '\r');
    if (c != '\n' && c != EOF) ungetc(c, fin);

    // Allocate merkle branch storage
    char *raw_merkle = new char[tx * 65];
    char **branches = new char*[tx];
    bool branch_ok = true;
    for (int i = 0; i < tx; ++i) {
        branches[i] = raw_merkle + i * 65;
        if (fscanf(fin, "%64s", branches[i]) != 1) {
            fprintf(stderr, "Failed to read merkle branch %d\n", i);
            branch_ok = false;
            break;
        }
        branches[i][64] = '\0';
    }
    if (!branch_ok) {
        fprintf(fout, "ffffffff\n");
        delete[] branches;
        delete[] raw_merkle;
        return;
    }

    // Calculate merkle root
    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, branches);

    printf("merkle root(little): "); print_hex(merkle_root, 32); printf("\n");
    printf("merkle root(big):    "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("Block info (big):\n");
    printf("  version:   %s\n", version);
    printf("  prevhash:  %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  ntime:     %s\n", ntime);
    printf("  nbits:     %s\n", nbits);
    printf("  tx:        %d\n", tx);
    printf("  nonce:     ???\n\n");

    // Build block (little-endian fields)
    HashBlock block{};
    convert_string_to_little_endian_bytes((unsigned char*)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char*)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char*)&block.ntime, ntime, 8);
    block.nonce = 0;

    // Target
    unsigned char target_hex[32]; memset(target_hex, 0, 32);
    compute_target(target_hex, block.nbits);
    printf("Target value (big):  "); print_hex_inverse(target_hex, 32); printf("\n");

    // GPU mine
    unsigned int found_nonce = 0;
    unsigned char found_hash[32];
    bool ok = gpu_mine(block, target_hex, found_nonce, found_hash, 0x100000000ULL);

    // Fallback (limited)
    if (!ok) {
        printf("GPU no solution, fallback scanning first 5M nonces.\n");
        SHA256 ctx; bool cpu_found = false;
        for (block.nonce = 0; block.nonce < 5000000; ++block.nonce) {
            double_sha256(&ctx, (unsigned char*)&block, sizeof(block));
            if (little_endian_bit_comparison(ctx.b, target_hex, 32) < 0) {
                found_nonce = block.nonce;
                memcpy(found_hash, ctx.b, 32);
                cpu_found = true;
                break;
            }
        }
        if (!cpu_found) {
            fprintf(stderr, "Fallback failed.\n");
            fprintf(fout, "ffffffff\n");
            delete[] branches;
            delete[] raw_merkle;
            return;
        }
    }

    block.nonce = found_nonce;
    printf("Found Solution!! Nonce: %u\n", found_nonce);
    printf("hash(little): "); print_hex(found_hash, 32); printf("\n");
    printf("hash(big):    "); print_hex_inverse(found_hash, 32); printf("\n\n");

    for (int i = 0; i < 4; ++i)
        fprintf(fout, "%02x", ((unsigned char*)&found_nonce)[i]);
    fprintf(fout, "\n");

    delete[] branches;
    delete[] raw_merkle;
}

// ------------------ main ------------------
int main(int argc, char **argv) {
    if(argc != 3) {
        fprintf(stderr,"usage: hw4 <in> <out>\n");
        return 1;
    }
    sha256_init_constants();

    FILE *fin = fopen(argv[1], "r");
    if(!fin) { fprintf(stderr,"Cannot open input %s\n", argv[1]); return 1; }
    FILE *fout = fopen(argv[2], "w");
    if(!fout){ fprintf(stderr,"Cannot open output %s\n", argv[2]); fclose(fin); return 1; }

    int totalblock = 0;
    if(fscanf(fin, "%d\n", &totalblock) != 1 || totalblock <= 0) {
        fprintf(stderr,"Bad block count\n");
        fclose(fin); fclose(fout); return 1;
    }
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i) {
        printf("=== Solving block %d/%d ===\n", i+1, totalblock);
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}