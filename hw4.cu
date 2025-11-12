//***********************************************************************************
// GPU Mining Version (rewritten from sequential reference)
//***********************************************************************************

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
__device__ unsigned int d_solution_nonce;
__device__ int d_found;
__device__ unsigned char d_solution_hash[32];

// New constants for optimized mining
__constant__ WORD d_midstate[8];          // midstate after first 64 bytes
__constant__ unsigned char d_tail_fixed[12]; // bytes 64..75 (last 16 bytes = 12 fixed + 4 nonce)

// ------------------ Kernel ------------------
__global__ void mine_kernel(HashBlock base, unsigned long long max_nonce_space) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
    for(unsigned long long nonce = tid; nonce < max_nonce_space; nonce += total_threads) {
        if(atomicAdd(&d_found, 0) != 0) return;
        HashBlock local = base;
        local.nonce = (unsigned int)nonce;
        SHA256 ctx;
        double_sha256(&ctx, (unsigned char*)&local, sizeof(HashBlock));
        if(little_endian_bit_comparison(ctx.b, d_target, 32) < 0) {
            if(atomicCAS(&d_found, 0, 1) == 0) {
                d_solution_nonce = (unsigned int)nonce;
                for(int i=0;i<32;++i) d_solution_hash[i] = ctx.b[i];
            }
            return;
        }
    }
}

// Optimized kernel: use midstate and only process second chunk
__global__ void mine_kernel_opt(unsigned long long max_nonce_space,
                                unsigned long long total_threads,
                                unsigned char *out_hash) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    for(unsigned long long nonce = tid; nonce < max_nonce_space; nonce += total_threads) {
        if(atomicAdd(&d_found, 0) != 0) return;

        // Build second 512-bit chunk (64 bytes)
        unsigned char chunk[64];
        // bytes 0..11 fixed tail
        #pragma unroll
        for(int i=0;i<12;++i) chunk[i] = d_tail_fixed[i];
        // bytes 12..15 = nonce (little-endian as in header)
        unsigned int n = (unsigned int)nonce;
        chunk[12] = (unsigned char)(n & 0xff);
        chunk[13] = (unsigned char)((n >> 8) & 0xff);
        chunk[14] = (unsigned char)((n >> 16) & 0xff);
        chunk[15] = (unsigned char)((n >> 24) & 0xff);
        // padding
        chunk[16] = 0x80;
        for(int i=17;i<56;++i) chunk[i] = 0x00;
        // message length (80 bytes = 640 bits) big-endian per existing implementation pattern
        unsigned long long L = 640ULL;
        chunk[63] = (unsigned char)(L);
        chunk[62] = (unsigned char)(L >> 8);
        chunk[61] = (unsigned char)(L >> 16);
        chunk[60] = (unsigned char)(L >> 24);
        chunk[59] = (unsigned char)(L >> 32);
        chunk[58] = (unsigned char)(L >> 40);
        chunk[57] = (unsigned char)(L >> 48);
        chunk[56] = (unsigned char)(L >> 56);

        // First hash (complete) from midstate + this chunk
        SHA256 ctx1;
        sha256_transform_from_state(d_midstate, chunk, &ctx1);

        // Second hash (standard)
        SHA256 ctx2;
        sha256(&ctx2, ctx1.b, 32);

        if(little_endian_bit_comparison(ctx2.b, d_target, 32) < 0) {
            if(atomicCAS(&d_found, 0, 1) == 0) {
                d_solution_nonce = (unsigned int)nonce;
                for(int i=0;i<32;++i) d_solution_hash[i] = ctx2.b[i];
                // Optionally copy out hash for debug
                if(out_hash){
                    for(int i=0;i<32;++i) out_hash[i] = ctx2.b[i];
                }
            }
            return;
        }
    }
}

// ------------------ Target Computation ------------------
void compute_target(unsigned char target_hex[32], unsigned int nbits_le) {
    // Match the sequential implementation exactly
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
bool gpu_mine(HashBlock &block, const unsigned char target_hex[32],
              unsigned int &out_nonce, unsigned char out_hash[32],
              unsigned long long max_nonce_space) {

    // Prepare 80-byte header (Bitcoin-like)
    unsigned char header[80];
    // layout: version(4) + prevhash(32) + merkle(32) + ntime(4) + nbits(4) + nonce(4)
    memcpy(header, &block.version, 4);
    memcpy(header+4,  block.prevhash, 32);
    memcpy(header+36, block.merkle_root, 32);
    memcpy(header+68, &block.ntime, 4);
    memcpy(header+72, &block.nbits, 4);
    // nonce left zero for midstate (inserted in kernel)
    memset(header+76, 0, 4);

    // Compute midstate of first 64 bytes
    WORD midstate_host[8];
    sha256_compute_midstate(header, midstate_host);

    // Upload target, midstate, tail (bytes 64..75)
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_target, target_hex, 32));
    int zero = 0;
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_found, &zero, sizeof(int)));
    unsigned int init_nonce = 0;
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_solution_nonce, &init_nonce, sizeof(unsigned int)));
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_midstate, midstate_host, sizeof(midstate_host)));
    CUDA_CALL_OR_FALSE(cudaMemcpyToSymbol(d_tail_fixed, header+64, 12));

    // Launch optimized kernel
    dim3 threads(256);
    dim3 blocks(512);
    unsigned long long total_threads = (unsigned long long)threads.x * blocks.x;
    mine_kernel_opt<<<blocks, threads>>>(max_nonce_space, total_threads, nullptr);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr,"Kernel failure\n");
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